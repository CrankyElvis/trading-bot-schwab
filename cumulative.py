"""
Cumulative validation report. v2.

Walks validation/history/*.json, builds rolling and since-inception metrics
showing how the live paper bot has tracked the backtester since day 1.

v2 fixes over v1:
- Signed P&L delta is primary metric; realization ratio guarded to bt > 0
- Score drift uses Kolmogorov-Smirnov two-sample test (or skips if scipy unavailable)
- Cross-day trade dedup via (ticker, trade_date) key — re-reconciled days
  don't get double-counted in the cumulative roll-up
- sort_time uses proper datetime parsing, not lex-sort on mixed-format strings
- Open trades excluded from P&L stats but counted in agreement
- VALIDATION_DIR fallback if not in bot.config

Usage:
    python -m validation.cumulative
    python -m validation.cumulative --html-only
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from bot.config import VALIDATION_DIR
except ImportError:
    VALIDATION_DIR = "/opt/bot/validation"

try:
    from scipy.stats import ks_2samp
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


log = logging.getLogger("cumulative")

HISTORY_DIR = Path(VALIDATION_DIR) / "history"
OUTPUT_DIR = Path(VALIDATION_DIR) / "cumulative"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROLLING_WINDOW = 20             # matches SAFETY #5 tripwire windows
KS_PVALUE_THRESHOLD = 0.01      # score drift trigger
MIN_TRADES_FOR_KS = 30


@dataclass
class CumulativeReport:
    inception_date: str
    last_date: str
    trading_days: int

    # Trade counts
    total_live_trades: int
    total_bt_trades: int
    total_matched: int
    total_closed_matched: int
    overall_agreement_rate: float

    # P&L since inception — closed trades only
    live_total_pnl: float
    bt_total_pnl: float
    pnl_delta_total: float                       # PRIMARY: signed dollars
    pnl_realization_ratio: Optional[float]       # only when bt > 0

    # Slippage
    median_entry_slippage_bps: Optional[float]
    median_exit_slippage_bps: Optional[float]

    # Win rates
    live_win_rate: Optional[float]
    bt_win_rate: Optional[float]

    # Rolling-20 tripwires (current state)
    rolling_win_rate_delta_pp: Optional[float]
    rolling_pnl_delta: Optional[float]           # signed dollars, last 20 matched closed
    rolling_pnl_realization: Optional[float]     # ratio, only when rolling bt > 0
    tripwire_rolling_win_rate: bool
    tripwire_rolling_pnl_underperform: bool
    tripwire_score_distribution_drift: bool
    ks_pvalue: Optional[float]                   # for transparency

    # Per-instrument
    equity_pnl_delta_total: float
    csp_pnl_delta_total: float

    # Daily series for HTML chart
    daily_series: list[dict]


def load_all_daily_reports() -> list[dict]:
    """Load every daily report, sorted by trade_date ascending."""
    files = sorted(HISTORY_DIR.glob("*.json"))
    out = []
    for f in files:
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            log.exception("Could not parse %s", f)
    return out


def flatten_trades_dedup(daily_reports: list[dict]) -> pd.DataFrame:
    """Flatten trade_deltas across all daily reports, deduplicating by
    (ticker, trade_date, source). A multi-day position appears in its
    entry-date report on every reconcile run; we keep ONLY the latest
    version (from the most-recently-generated daily report)."""
    rows = []
    for daily in daily_reports:
        generated_at = daily.get("generated_at", "")
        for td in daily["trade_deltas"]:
            rows.append({**td, "_generated_at": generated_at})
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Dedup: a trade is uniquely identified by (ticker, trade_date, source).
    # If the same trade appears in multiple daily reports (because past dates
    # were re-reconciled to pick up closures), keep the latest one.
    df = df.sort_values("_generated_at").drop_duplicates(
        subset=["ticker", "trade_date", "source"], keep="last"
    )

    # Order chronologically by entry time for rolling windows
    df["_sort_time"] = pd.to_datetime(
        df["live_entry_time"].fillna(df["bt_entry_time"]),
        utc=True, errors="coerce"
    )
    df = df.sort_values("_sort_time").reset_index(drop=True)
    return df.drop(columns=["_generated_at", "_sort_time"])


def build_cumulative(daily_reports: list[dict]) -> CumulativeReport:
    if not daily_reports:
        raise ValueError("No daily reports yet — has paper trading started?")

    trades = flatten_trades_dedup(daily_reports)
    if trades.empty:
        log.warning("Daily reports exist but contain no trades")

    matched = trades[trades["matched"] == True] if not trades.empty else trades  # noqa: E712
    closed_matched = (matched[matched["is_open"] == False]  # noqa: E712
                      if not matched.empty else matched)
    closed_all = (trades[trades["is_open"] == False]  # noqa: E712
                  if not trades.empty else trades)

    # P&L over closed trades only
    if not closed_all.empty:
        total_live_pnl = float(closed_all["live_pnl"].fillna(0).sum())
        total_bt_pnl = float(closed_all["bt_pnl"].fillna(0).sum())
    else:
        total_live_pnl = total_bt_pnl = 0.0

    pnl_delta_total = total_live_pnl - total_bt_pnl
    realization = (total_live_pnl / total_bt_pnl) if total_bt_pnl > 0 else None

    # Rolling-20 on most recent matched CLOSED trades
    rolling_win_rate_delta = None
    rolling_pnl_delta = None
    rolling_pnl_realization = None
    tripwire_rolling_win_rate = False
    tripwire_rolling_pnl_underperform = False

    if len(closed_matched) >= ROLLING_WINDOW:
        recent = closed_matched.tail(ROLLING_WINDOW)
        live_wr = float((recent["live_pnl"].fillna(0) > 0).mean())
        bt_wr = float((recent["bt_pnl"].fillna(0) > 0).mean())
        rolling_win_rate_delta = (live_wr - bt_wr) * 100
        tripwire_rolling_win_rate = rolling_win_rate_delta < -15

        recent_live_pnl = float(recent["live_pnl"].fillna(0).sum())
        recent_bt_pnl = float(recent["bt_pnl"].fillna(0).sum())
        rolling_pnl_delta = recent_live_pnl - recent_bt_pnl
        if recent_bt_pnl > 0:
            rolling_pnl_realization = recent_live_pnl / recent_bt_pnl
            tripwire_rolling_pnl_underperform = rolling_pnl_realization < 0.50
        # If recent_bt_pnl <= 0, the ratio is meaningless. Use signed delta
        # alone as the signal: live underperforming bt by more than half the
        # absolute bt P&L is the equivalent threshold.
        elif recent_bt_pnl < 0 and recent_live_pnl < recent_bt_pnl:
            tripwire_rolling_pnl_underperform = (
                abs(rolling_pnl_delta) > abs(recent_bt_pnl) * 0.5
            )

    # Score distribution drift — proper KS test
    score_drift = False
    ks_pvalue = None
    if HAVE_SCIPY and len(closed_matched) >= MIN_TRADES_FOR_KS:
        live_scores = closed_matched["live_score"].dropna().astype(float)
        bt_scores = closed_matched["bt_score"].dropna().astype(float)
        if len(live_scores) >= 10 and len(bt_scores) >= 10:
            try:
                stat, pval = ks_2samp(live_scores, bt_scores)
                ks_pvalue = float(pval)
                score_drift = pval < KS_PVALUE_THRESHOLD
            except Exception:
                log.exception("KS test failed")

    # Daily series for HTML
    daily_series = [
        {
            "date": r["trade_date"],
            "live_pnl": r["live_total_pnl"],
            "bt_pnl": r["bt_total_pnl"],
            "pnl_delta": r["pnl_delta"],
            "agreement": r["agreement_rate"],
            "n_matched": r["n_matched"],
            "n_still_open": r.get("n_still_open", 0),
        }
        for r in daily_reports
    ]

    # Overall counts — from deduplicated trades
    if not trades.empty:
        only_live_n = int((trades["source"] == "only_live").sum())
        only_bt_n = int((trades["source"] == "only_backtest").sum())
        matched_n = int(len(matched))
        denom = matched_n + only_live_n + only_bt_n
        overall_agreement = (matched_n / denom) if denom else 0.0
    else:
        only_live_n = only_bt_n = matched_n = 0
        overall_agreement = 0.0

    # Win rates over all closed trades
    if not closed_all.empty:
        live_wins_total = int((closed_all["live_pnl"].fillna(0) > 0).sum())
        bt_wins_total = int((closed_all["bt_pnl"].fillna(0) > 0).sum())
        live_closed = int(closed_all["live_pnl"].notna().sum())
        bt_closed = int(closed_all["bt_pnl"].notna().sum())
        live_win_rate = (live_wins_total / live_closed) if live_closed else None
        bt_win_rate = (bt_wins_total / bt_closed) if bt_closed else None
    else:
        live_win_rate = bt_win_rate = None

    # Slippage medians
    if not closed_matched.empty:
        entry_slips = closed_matched["entry_slippage_bps"].dropna()
        exit_slips = closed_matched["exit_slippage_bps"].dropna()
        median_entry = float(entry_slips.median()) if len(entry_slips) else None
        median_exit = float(exit_slips.median()) if len(exit_slips) else None
    else:
        median_entry = median_exit = None

    # Per-instrument
    if not closed_matched.empty:
        eq = closed_matched[closed_matched["instrument"] == "equity"]
        csp = closed_matched[closed_matched["instrument"] == "csp"]
        equity_delta = float(eq["pnl_delta"].fillna(0).sum())
        csp_delta = float(csp["pnl_delta"].fillna(0).sum())
    else:
        equity_delta = csp_delta = 0.0

    return CumulativeReport(
        inception_date=daily_reports[0]["trade_date"],
        last_date=daily_reports[-1]["trade_date"],
        trading_days=len(daily_reports),
        total_live_trades=matched_n + only_live_n,
        total_bt_trades=matched_n + only_bt_n,
        total_matched=matched_n,
        total_closed_matched=int(len(closed_matched)) if not closed_matched.empty else 0,
        overall_agreement_rate=overall_agreement,
        live_total_pnl=total_live_pnl,
        bt_total_pnl=total_bt_pnl,
        pnl_delta_total=pnl_delta_total,
        pnl_realization_ratio=realization,
        median_entry_slippage_bps=median_entry,
        median_exit_slippage_bps=median_exit,
        live_win_rate=live_win_rate,
        bt_win_rate=bt_win_rate,
        rolling_win_rate_delta_pp=rolling_win_rate_delta,
        rolling_pnl_delta=rolling_pnl_delta,
        rolling_pnl_realization=rolling_pnl_realization,
        tripwire_rolling_win_rate=tripwire_rolling_win_rate,
        tripwire_rolling_pnl_underperform=tripwire_rolling_pnl_underperform,
        tripwire_score_distribution_drift=score_drift,
        ks_pvalue=ks_pvalue,
        equity_pnl_delta_total=equity_delta,
        csp_pnl_delta_total=csp_delta,
        daily_series=daily_series,
    )


def render_html(report: CumulativeReport) -> str:
    rows = []
    for d in report.daily_series:
        delta = d["pnl_delta"]
        cls = "green" if delta >= 0 else "red"
        open_note = f" ({d['n_still_open']} open)" if d.get("n_still_open") else ""
        rows.append(
            f'<tr><td>{d["date"]}</td>'
            f'<td>{d["n_matched"]}{open_note}</td>'
            f'<td>{d["agreement"]:.0%}</td>'
            f'<td>${d["live_pnl"]:,.2f}</td>'
            f'<td>${d["bt_pnl"]:,.2f}</td>'
            f'<td class="{cls}">${delta:+,.2f}</td></tr>'
        )

    tripwires = []
    if report.tripwire_rolling_win_rate:
        tripwires.append(
            f"⚠ Rolling-20 win rate {report.rolling_win_rate_delta_pp:+.1f}pp "
            f"vs backtest (threshold: -15pp)"
        )
    if report.tripwire_rolling_pnl_underperform:
        tripwires.append(
            f"⚠ Rolling-20 P&L delta ${report.rolling_pnl_delta:+,.0f} — "
            f"live materially underperforming backtest"
        )
    if report.tripwire_score_distribution_drift:
        tripwires.append(
            f"⚠ Score distribution drift detected (KS p={report.ks_pvalue:.4f})"
        )
    tripwire_html = (
        "<div class='alert'>" + "<br>".join(tripwires) + "</div>"
        if tripwires else
        "<div class='ok'>✓ No tripwires triggered</div>"
    )

    realization = (
        f"{report.pnl_realization_ratio:.1%}"
        if report.pnl_realization_ratio is not None else "n/a"
    )
    delta_cls = "green" if report.pnl_delta_total >= 0 else "red"

    scipy_note = (
        "" if HAVE_SCIPY else
        "<p style='color:#999;font-size:0.85em'>"
        "Note: scipy not installed — score drift tripwire disabled. "
        "<code>pip install scipy</code> to enable.</p>"
    )

    return f"""<!DOCTYPE html>
<html><head><style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 20px auto; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid #ddd; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.green {{ color: #1a7f3c; }}
.red {{ color: #c62828; }}
.alert {{ background: #fff3cd; padding: 12px; border-left: 4px solid #f57c00; margin: 10px 0; }}
.ok {{ background: #e8f5e9; padding: 12px; border-left: 4px solid #2e7d32; margin: 10px 0; }}
.kpi {{ display: inline-block; padding: 8px 16px; margin: 4px; background: #f5f5f5; border-radius: 4px; }}
.kpi b {{ display: block; font-size: 1.4em; }}
</style></head><body>
<h1>Paper vs Backtest — Since Inception</h1>
<p>Inception: <b>{report.inception_date}</b> · Last: <b>{report.last_date}</b>
   · Trading days: <b>{report.trading_days}</b></p>

<div>
  <div class="kpi"><b>{report.overall_agreement_rate:.0%}</b>signal agreement</div>
  <div class="kpi"><b>${report.live_total_pnl:,.0f}</b>live P&L</div>
  <div class="kpi"><b>${report.bt_total_pnl:,.0f}</b>backtest P&L</div>
  <div class="kpi"><b class="{delta_cls}">${report.pnl_delta_total:+,.0f}</b>signed delta</div>
  <div class="kpi"><b>{realization}</b>realization (when bt&gt;0)</div>
  <div class="kpi"><b>{report.median_entry_slippage_bps or 0:.1f} bps</b>median entry slip</div>
</div>

{tripwire_html}

<h2>Daily breakdown</h2>
<table>
<tr><th>Date</th><th>Matched</th><th>Agreement</th>
    <th>Live P&L</th><th>Backtest P&L</th><th>Delta</th></tr>
{''.join(rows)}
</table>

{scipy_note}
</body></html>"""


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--html-only", action="store_true")
    args = ap.parse_args()

    daily = load_all_daily_reports()
    if not daily:
        log.error("No daily reports in %s — nothing to roll up", HISTORY_DIR)
        return

    report = build_cumulative(daily)

    if not args.html_only:
        json_path = OUTPUT_DIR / "cumulative.json"
        json_path.write_text(json.dumps(asdict(report), indent=2, default=str))
        log.info("Wrote %s", json_path)

    html_path = OUTPUT_DIR / "cumulative.html"
    html_path.write_text(render_html(report))
    log.info("Wrote %s", html_path)

    # Console summary
    print(f"\n=== Validation summary ({report.inception_date} → {report.last_date}) ===")
    print(f"Trading days:          {report.trading_days}")
    print(f"Signal agreement:      {report.overall_agreement_rate:.1%}")
    print(f"Live P&L (closed):     ${report.live_total_pnl:,.2f}")
    print(f"Backtest P&L (closed): ${report.bt_total_pnl:,.2f}")
    print(f"Signed delta:          ${report.pnl_delta_total:+,.2f}")
    if report.pnl_realization_ratio is not None:
        print(f"P&L realization:       {report.pnl_realization_ratio:.1%}")
    else:
        print(f"P&L realization:       n/a (backtest P&L <= 0)")
    if report.median_entry_slippage_bps is not None:
        print(f"Median entry slip:     {report.median_entry_slippage_bps:.1f} bps")
    if report.rolling_win_rate_delta_pp is not None:
        print(f"Rolling-20 WR delta:   {report.rolling_win_rate_delta_pp:+.1f} pp")
    if report.rolling_pnl_delta is not None:
        print(f"Rolling-20 PnL delta:  ${report.rolling_pnl_delta:+,.2f}")
    print(f"Equity delta:          ${report.equity_pnl_delta_total:+,.2f}")
    print(f"CSP delta:             ${report.csp_pnl_delta_total:+,.2f}")
    if report.ks_pvalue is not None:
        print(f"Score drift KS p-val:  {report.ks_pvalue:.4f}")
    if any([report.tripwire_rolling_win_rate,
            report.tripwire_rolling_pnl_underperform,
            report.tripwire_score_distribution_drift]):
        print("\n*** TRIPWIRES TRIGGERED — review before continuing ***")


if __name__ == "__main__":
    main()