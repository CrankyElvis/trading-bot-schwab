"""
Daily paper-vs-backtest reconciliation. v2.

Run nightly after market close. Pulls the day's paper trades from the bot's
trade log, runs the backtester over the same date with the same data feeds,
joins on (ticker, date) and emits per-trade deltas + aggregate metrics.

Persists results to validation/history/ so cumulative drift can be tracked
across the entire paper trading window.

v2 fixes over v1:
- Signed P&L delta is now the primary metric (avoids negative-bt-pnl sign flip)
- _safe_iso handles strings, datetimes, and pd.Timestamp uniformly
- Multi-day positions handled via merge-on-persist (entry-date keyed)
- VALIDATION_DIR has a fallback if not in bot.config
- Backtester invoked with a fixed seed for reproducibility
- match_trades resets bt index defensively before iterating
- Score drift uses Kolmogorov-Smirnov two-sample test
- Subject-line P&L realization guards against meaningless ratios

Usage:
    python -m validation.reconcile --date 2026-05-23
    python -m validation.reconcile
    python -m validation.reconcile --backfill 2026-05-01 2026-05-23
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Any

import pandas as pd

# Project imports — assumed to exist in the trading-bot-schwab repo
from bot.storage import load_paper_trades       # see contract below
from backtester.engine import run_backtest      # existing backtester entry point

# VALIDATION_DIR fallback — add to bot/config.py: VALIDATION_DIR = "/opt/bot/validation"
try:
    from bot.config import VALIDATION_DIR
except ImportError:
    VALIDATION_DIR = "/opt/bot/validation"


log = logging.getLogger("reconcile")

HISTORY_DIR = Path(VALIDATION_DIR) / "history"
SNAPSHOT_DIR = Path(VALIDATION_DIR) / "snapshots"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# Trades within this many seconds count as the "same" entry
TIMESTAMP_TOLERANCE_SEC = 120

# Fixed seed for backtester reproducibility — same seed every reconcile run
BACKTEST_SEED = 42

# What the backtester models for slippage. Update if the backtester changes.
BACKTEST_SLIPPAGE_ASSUMPTION_BPS = 5.0


# ---------------------------------------------------------------------------
# Contract for load_paper_trades(entry_date_start, entry_date_end)
# ---------------------------------------------------------------------------
# MUST filter by ENTRY DATE (not exit date, not "active during range").
# A trade entered 2026-05-23 stays attached to the 2026-05-23 report forever,
# regardless of when it closes. When the trade later closes (e.g. CSP closes
# 2026-06-23), the *same* row is returned by load_paper_trades for the
# original entry date — now with exit_price / exit_time / exit_reason filled.
#
# Required columns:
#   ticker, entry_time, entry_price, exit_time, exit_price, pnl, exit_reason,
#   score, regime, signals_fired (JSON list or list), instrument
#
# exit_time / exit_price / exit_reason / pnl may be None/NaN for open trades.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TradeDelta:
    """Per-trade comparison. Joined on (ticker, entry_date) within timestamp tolerance."""
    ticker: str
    trade_date: str                # entry date — stable identity across reruns
    matched: bool
    source: str                    # "both" | "only_live" | "only_backtest"
    is_open: bool                  # True if either side has no exit yet

    # Entry
    live_entry_time: Optional[str] = None
    bt_entry_time: Optional[str] = None
    live_entry_price: Optional[float] = None
    bt_entry_price: Optional[float] = None
    entry_slippage_bps: Optional[float] = None

    # Exit (may be None until trade closes)
    live_exit_time: Optional[str] = None
    bt_exit_time: Optional[str] = None
    live_exit_price: Optional[float] = None
    bt_exit_price: Optional[float] = None
    exit_slippage_bps: Optional[float] = None
    live_exit_reason: Optional[str] = None
    bt_exit_reason: Optional[str] = None

    # P&L (None until closed)
    live_pnl: Optional[float] = None
    bt_pnl: Optional[float] = None
    pnl_delta: Optional[float] = None    # live - bt, dollars (always meaningful)

    # Context
    live_score: Optional[float] = None
    bt_score: Optional[float] = None
    regime: Optional[str] = None
    signals_fired: list[str] = field(default_factory=list)
    instrument: str = "equity"           # "equity" | "csp"


@dataclass
class DailyReport:
    trade_date: str
    generated_at: str

    # Counts (closed + open)
    n_live: int
    n_backtest: int
    n_matched: int
    n_only_live: int
    n_only_backtest: int
    n_still_open: int                    # exit not yet known on either side
    agreement_rate: float                # matched / (matched + only_live + only_bt)

    # P&L — only closed trades count
    live_total_pnl: float
    bt_total_pnl: float
    pnl_delta: float                     # PRIMARY METRIC: signed dollar delta
    pnl_realization_ratio: Optional[float]   # live/bt, ONLY when bt > 0; else None

    # Slippage (matched closed trades only)
    median_entry_slippage_bps: Optional[float]
    median_exit_slippage_bps: Optional[float]
    p95_entry_slippage_bps: Optional[float]
    p95_exit_slippage_bps: Optional[float]

    # Win rates
    live_win_rate: Optional[float]
    bt_win_rate: Optional[float]
    win_rate_delta_pp: Optional[float]

    # Per-instrument signed dollar deltas
    equity_pnl_delta: float
    csp_pnl_delta: float

    # Tripwires
    tripwire_agreement_below_70pct: bool
    tripwire_pnl_delta_negative_large: bool   # signed: live underperformed bt by >$X
    tripwire_slippage_2x_assumption: bool

    trade_deltas: list[dict] = field(default_factory=list)


# Threshold for tripwire_pnl_delta_negative_large — daily-level
DAILY_PNL_UNDERPERFORM_USD = 500.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_live_trades(trade_date: date) -> pd.DataFrame:
    """Paper trades whose ENTRY DATE == trade_date.

    Returns trades regardless of whether they've closed yet. Exit columns
    may be NaN for open positions.
    """
    df = load_paper_trades(entry_date_start=trade_date, entry_date_end=trade_date)
    if df.empty:
        log.warning("No live paper trades for %s", trade_date)
        return df

    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")

    # signals_fired may come as JSON string from SQLite TEXT
    if "signals_fired" in df.columns and len(df):
        sample = df["signals_fired"].iloc[0]
        if isinstance(sample, str):
            df["signals_fired"] = df["signals_fired"].apply(
                lambda s: json.loads(s) if isinstance(s, str) else (s or [])
            )

    return df.reset_index(drop=True)


def load_backtest_trades(trade_date: date) -> pd.DataFrame:
    """Run the backtester for a single day against the archived data snapshot.

    Critical: the snapshot must contain *exactly* the data the live bot saw
    that day — no future revisions to UW flow, no updated earnings, no
    repriced quotes. Bot must archive snapshots nightly to enable this.
    """
    snapshot_path = SNAPSHOT_DIR / f"{trade_date}.parquet"
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"No data snapshot for {trade_date} at {snapshot_path}. "
            f"Bot must archive daily snapshots to {SNAPSHOT_DIR}/ for "
            f"reconciliation to produce honest results."
        )

    result = run_backtest(
        start_date=trade_date,
        end_date=trade_date,
        data_snapshot=snapshot_path,
        config_overrides=None,
        seed=BACKTEST_SEED,
    )
    df = result.trade_log.copy()
    if df.empty:
        return df

    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_trades(live: pd.DataFrame, bt: pd.DataFrame) -> list[TradeDelta]:
    """Join live and backtest trades by (ticker, entry_date) within tolerance.

    Pure positional matching on reset indices, so we own the index semantics
    rather than inheriting whatever the storage/backtester layers produce.
    """
    deltas: list[TradeDelta] = []
    live = live.reset_index(drop=True) if not live.empty else live
    bt = bt.reset_index(drop=True) if not bt.empty else bt
    bt_used: set[int] = set()

    if not live.empty:
        for i, lrow in live.iterrows():
            if bt.empty:
                deltas.append(_only_live(lrow))
                continue

            mask = (bt["ticker"] == lrow["ticker"]) & (~bt.index.isin(bt_used))
            candidates = bt[mask]
            if candidates.empty:
                deltas.append(_only_live(lrow))
                continue

            time_diffs = (candidates["entry_time"] - lrow["entry_time"]).abs()
            min_idx = time_diffs.idxmin()
            min_diff = time_diffs.loc[min_idx]

            if pd.isna(min_diff) or min_diff.total_seconds() > TIMESTAMP_TOLERANCE_SEC:
                deltas.append(_only_live(lrow))
                continue

            bt_used.add(min_idx)
            deltas.append(_matched(lrow, bt.loc[min_idx]))

    if not bt.empty:
        for idx, brow in bt.iterrows():
            if idx not in bt_used:
                deltas.append(_only_backtest(brow))

    return deltas


def _matched(live_row, bt_row) -> TradeDelta:
    live_exit_price = _safe_float(live_row.get("exit_price"))
    bt_exit_price = _safe_float(bt_row.get("exit_price"))
    live_pnl = _safe_float(live_row.get("pnl"))
    bt_pnl = _safe_float(bt_row.get("pnl"))

    is_open = live_exit_price is None or bt_exit_price is None

    pnl_delta = None
    if live_pnl is not None and bt_pnl is not None:
        pnl_delta = live_pnl - bt_pnl

    return TradeDelta(
        ticker=str(live_row["ticker"]),
        trade_date=_extract_date_str(live_row["entry_time"]),
        matched=True,
        source="both",
        is_open=is_open,
        live_entry_time=_safe_iso(live_row["entry_time"]),
        bt_entry_time=_safe_iso(bt_row["entry_time"]),
        live_entry_price=_safe_float(live_row["entry_price"]),
        bt_entry_price=_safe_float(bt_row["entry_price"]),
        entry_slippage_bps=_bps(live_row.get("entry_price"), bt_row.get("entry_price")),
        live_exit_time=_safe_iso(live_row.get("exit_time")),
        bt_exit_time=_safe_iso(bt_row.get("exit_time")),
        live_exit_price=live_exit_price,
        bt_exit_price=bt_exit_price,
        exit_slippage_bps=_bps(live_row.get("exit_price"), bt_row.get("exit_price")),
        live_exit_reason=_safe_str(live_row.get("exit_reason")),
        bt_exit_reason=_safe_str(bt_row.get("exit_reason")),
        live_pnl=live_pnl,
        bt_pnl=bt_pnl,
        pnl_delta=pnl_delta,
        live_score=_safe_float(live_row.get("score")),
        bt_score=_safe_float(bt_row.get("score")),
        regime=_safe_str(live_row.get("regime")),
        signals_fired=list(live_row.get("signals_fired") or []),
        instrument=str(live_row.get("instrument", "equity")),
    )


def _only_live(row) -> TradeDelta:
    exit_price = _safe_float(row.get("exit_price"))
    return TradeDelta(
        ticker=str(row["ticker"]),
        trade_date=_extract_date_str(row["entry_time"]),
        matched=False,
        source="only_live",
        is_open=exit_price is None,
        live_entry_time=_safe_iso(row["entry_time"]),
        live_entry_price=_safe_float(row["entry_price"]),
        live_exit_time=_safe_iso(row.get("exit_time")),
        live_exit_price=exit_price,
        live_exit_reason=_safe_str(row.get("exit_reason")),
        live_pnl=_safe_float(row.get("pnl")),
        live_score=_safe_float(row.get("score")),
        regime=_safe_str(row.get("regime")),
        signals_fired=list(row.get("signals_fired") or []),
        instrument=str(row.get("instrument", "equity")),
    )


def _only_backtest(row) -> TradeDelta:
    exit_price = _safe_float(row.get("exit_price"))
    return TradeDelta(
        ticker=str(row["ticker"]),
        trade_date=_extract_date_str(row["entry_time"]),
        matched=False,
        source="only_backtest",
        is_open=exit_price is None,
        bt_entry_time=_safe_iso(row["entry_time"]),
        bt_entry_price=_safe_float(row["entry_price"]),
        bt_exit_time=_safe_iso(row.get("exit_time")),
        bt_exit_price=exit_price,
        bt_exit_reason=_safe_str(row.get("exit_reason")),
        bt_pnl=_safe_float(row.get("pnl")),
        bt_score=_safe_float(row.get("score")),
        regime=_safe_str(row.get("regime")),
        instrument=str(row.get("instrument", "equity")),
    )


# ---------------------------------------------------------------------------
# Helpers — safe across str/datetime/Timestamp/None/NaN
# ---------------------------------------------------------------------------

def _bps(live_p: Any, bt_p: Any) -> Optional[float]:
    lp = _safe_float(live_p)
    bp = _safe_float(bt_p)
    if lp is None or bp is None or bp == 0:
        return None
    return (lp - bp) / bp * 10_000


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def _safe_iso(v: Any) -> Optional[str]:
    """Return ISO 8601 string from datetime/Timestamp/str/None. Robust to any."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, str):
        return v
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


def _extract_date_str(v: Any) -> str:
    """Extract YYYY-MM-DD from a timestamp-ish value. Used for entry_date keying."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v[:10]
    if hasattr(v, "date"):
        return v.date().isoformat()
    return str(v)[:10]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def build_report(trade_date: date, deltas: list[TradeDelta]) -> DailyReport:
    matched = [d for d in deltas if d.matched]
    only_live = [d for d in deltas if d.source == "only_live"]
    only_bt = [d for d in deltas if d.source == "only_backtest"]
    still_open = [d for d in deltas if d.is_open]

    # Closed trades only for P&L stats — open positions have None pnl
    closed = [d for d in deltas if not d.is_open]
    closed_matched = [d for d in matched if not d.is_open]

    n_live = len(matched) + len(only_live)
    n_bt = len(matched) + len(only_bt)
    total = len(matched) + len(only_live) + len(only_bt)
    agreement = len(matched) / total if total else 0.0

    live_pnl = sum((d.live_pnl or 0.0) for d in closed)
    bt_pnl = sum((d.bt_pnl or 0.0) for d in closed)
    pnl_delta = live_pnl - bt_pnl

    # Realization ratio is ONLY meaningful when bt_pnl > 0
    if bt_pnl > 0:
        pnl_realization_ratio = live_pnl / bt_pnl
    else:
        pnl_realization_ratio = None

    entry_slips = [d.entry_slippage_bps for d in closed_matched
                   if d.entry_slippage_bps is not None]
    exit_slips = [d.exit_slippage_bps for d in closed_matched
                  if d.exit_slippage_bps is not None]

    # Win rates over CLOSED trades on each side
    closed_live = [d for d in closed if d.live_pnl is not None]
    closed_bt = [d for d in closed if d.bt_pnl is not None]
    live_wins = sum(1 for d in closed_live if (d.live_pnl or 0) > 0)
    bt_wins = sum(1 for d in closed_bt if (d.bt_pnl or 0) > 0)
    live_wr = live_wins / len(closed_live) if closed_live else None
    bt_wr = bt_wins / len(closed_bt) if closed_bt else None

    equity_delta = sum((d.pnl_delta or 0.0) for d in closed_matched
                       if d.instrument == "equity")
    csp_delta = sum((d.pnl_delta or 0.0) for d in closed_matched
                    if d.instrument == "csp")

    median_entry = pd.Series(entry_slips).median() if entry_slips else None
    median_exit = pd.Series(exit_slips).median() if exit_slips else None
    p95_entry = pd.Series(entry_slips).quantile(0.95) if entry_slips else None
    p95_exit = pd.Series(exit_slips).quantile(0.95) if exit_slips else None

    return DailyReport(
        trade_date=str(trade_date),
        generated_at=datetime.now(timezone.utc).isoformat(),
        n_live=n_live,
        n_backtest=n_bt,
        n_matched=len(matched),
        n_only_live=len(only_live),
        n_only_backtest=len(only_bt),
        n_still_open=len(still_open),
        agreement_rate=agreement,
        live_total_pnl=live_pnl,
        bt_total_pnl=bt_pnl,
        pnl_delta=pnl_delta,
        pnl_realization_ratio=pnl_realization_ratio,
        median_entry_slippage_bps=median_entry,
        median_exit_slippage_bps=median_exit,
        p95_entry_slippage_bps=p95_entry,
        p95_exit_slippage_bps=p95_exit,
        live_win_rate=live_wr,
        bt_win_rate=bt_wr,
        win_rate_delta_pp=((live_wr - bt_wr) * 100
                          if (live_wr is not None and bt_wr is not None) else None),
        equity_pnl_delta=equity_delta,
        csp_pnl_delta=csp_delta,
        tripwire_agreement_below_70pct=agreement < 0.70 and total >= 5,
        tripwire_pnl_delta_negative_large=pnl_delta < -DAILY_PNL_UNDERPERFORM_USD,
        tripwire_slippage_2x_assumption=(
            median_entry is not None
            and abs(median_entry) > 2 * BACKTEST_SLIPPAGE_ASSUMPTION_BPS
        ),
        trade_deltas=[asdict(d) for d in deltas],
    )


# ---------------------------------------------------------------------------
# Persistence — merge-not-overwrite for multi-day position handling
# ---------------------------------------------------------------------------

def persist_report(report: DailyReport) -> Path:
    """Write report JSON. Idempotent — re-running for the same date overwrites,
    which is correct since the report reflects current state of all trades
    entered that day (closed or still open)."""
    path = HISTORY_DIR / f"{report.trade_date}.json"
    path.write_text(json.dumps(asdict(report), indent=2, default=str))
    log.info("Wrote %s (closed: %d, open: %d, agreement: %.1f%%)",
             path,
             report.n_matched + report.n_only_live + report.n_only_backtest - report.n_still_open,
             report.n_still_open,
             report.agreement_rate * 100)
    return path


def reconcile_day(trade_date: date) -> DailyReport:
    """Reconcile a single day. Re-runnable: re-reconciling a past date will
    pick up newly-closed multi-day positions and refresh their exit data."""
    log.info("Reconciling %s", trade_date)
    live = load_live_trades(trade_date)
    bt = load_backtest_trades(trade_date)
    deltas = match_trades(live, bt)
    report = build_report(trade_date, deltas)
    persist_report(report)
    return report


def reconcile_with_open_followups(trade_date: date, lookback_days: int = 60) -> DailyReport:
    """Reconcile today, then re-reconcile any past day that had open positions
    in case they've now closed. lookback_days bounds how far back to check —
    CSPs are 35-45 DTE so 60 is safe."""
    report = reconcile_day(trade_date)

    cutoff = trade_date - timedelta(days=lookback_days)
    for path in sorted(HISTORY_DIR.glob("*.json")):
        try:
            past_date = datetime.strptime(path.stem, "%Y-%m-%d").date()
        except ValueError:
            continue
        if past_date >= trade_date or past_date < cutoff:
            continue
        try:
            past = json.loads(path.read_text())
        except Exception:
            log.exception("Could not read %s", path)
            continue
        if past.get("n_still_open", 0) > 0:
            try:
                log.info("Re-reconciling %s (had %d open positions)",
                         past_date, past["n_still_open"])
                reconcile_day(past_date)
            except FileNotFoundError as e:
                log.warning("Skipping re-reconcile of %s: %s", past_date, e)
            except Exception:
                log.exception("Re-reconcile of %s failed", past_date)

    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date())
    ap.add_argument("--backfill", nargs=2, metavar=("START", "END"),
                    type=lambda s: datetime.strptime(s, "%Y-%m-%d").date())
    ap.add_argument("--no-followups", action="store_true",
                    help="Skip re-reconciling past dates with open positions")
    args = ap.parse_args()

    if args.backfill:
        start, end = args.backfill
        d = start
        while d <= end:
            if d.weekday() < 5:
                try:
                    reconcile_day(d)
                except FileNotFoundError as e:
                    log.warning("Skipping %s: %s", d, e)
                except Exception:
                    log.exception("Reconcile failed for %s", d)
            d += timedelta(days=1)
    else:
        target = args.date or date.today()
        if args.no_followups:
            reconcile_day(target)
        else:
            reconcile_with_open_followups(target)


if __name__ == "__main__":
    main()