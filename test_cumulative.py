"""
Test harness for v2 cumulative.py.

Builds a directory of synthetic daily reports and exercises the rollup logic.
"""

import sys
import types
import json
import shutil
from datetime import datetime, timedelta, date, timezone
from pathlib import Path
import tempfile

import pandas as pd


# ---------------------------------------------------------------------------
# Stub bot.config BEFORE importing cumulative
# ---------------------------------------------------------------------------

TMPDIR = Path(tempfile.mkdtemp())

bot_mod = types.ModuleType("bot")
bot_config = types.ModuleType("bot.config")
bot_config.VALIDATION_DIR = str(TMPDIR)
sys.modules["bot"] = bot_mod
sys.modules["bot.config"] = bot_config

sys.path.insert(0, str(Path(__file__).parent))
import cumulative  # noqa: E402


# Test utilities
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

results = []

def check(name, cond, detail=""):
    if cond:
        print(f"  {PASS} {name}")
        results.append(True)
    else:
        print(f"  {FAIL} {name} — {detail}")
        results.append(False)


def make_delta(ticker, trade_date, source="both", matched=True, is_open=False,
               live_pnl=None, bt_pnl=None, live_score=0.85, bt_score=0.85,
               instrument="equity",
               live_entry_time=None, bt_entry_time=None,
               entry_slip=None, exit_slip=None):
    """Construct a trade_delta dict matching the v2 dataclass shape."""
    if live_entry_time is None:
        live_entry_time = f"{trade_date}T09:35:00+00:00"
    if bt_entry_time is None:
        bt_entry_time = f"{trade_date}T09:35:10+00:00"

    pnl_delta = None
    if live_pnl is not None and bt_pnl is not None:
        pnl_delta = live_pnl - bt_pnl

    return {
        "ticker": ticker,
        "trade_date": trade_date,
        "matched": matched,
        "source": source,
        "is_open": is_open,
        "live_entry_time": live_entry_time if source != "only_backtest" else None,
        "bt_entry_time": bt_entry_time if source != "only_live" else None,
        "live_entry_price": 100.0 if source != "only_backtest" else None,
        "bt_entry_price": 100.0 if source != "only_live" else None,
        "entry_slippage_bps": entry_slip,
        "live_exit_time": f"{trade_date}T11:35:00+00:00" if not is_open and source != "only_backtest" else None,
        "bt_exit_time": f"{trade_date}T11:35:00+00:00" if not is_open and source != "only_live" else None,
        "live_exit_price": 105.0 if not is_open and source != "only_backtest" else None,
        "bt_exit_price": 105.0 if not is_open and source != "only_live" else None,
        "exit_slippage_bps": exit_slip,
        "live_exit_reason": "profit_target" if not is_open and source != "only_backtest" else None,
        "bt_exit_reason": "profit_target" if not is_open and source != "only_live" else None,
        "live_pnl": live_pnl,
        "bt_pnl": bt_pnl,
        "pnl_delta": pnl_delta,
        "live_score": live_score if source != "only_backtest" else None,
        "bt_score": bt_score if source != "only_live" else None,
        "regime": "neutral",
        "signals_fired": ["momentum"],
        "instrument": instrument,
    }


def make_daily_report(trade_date, deltas, generated_at=None):
    """Construct a daily report dict in v2 shape."""
    matched = [d for d in deltas if d["matched"]]
    only_live = [d for d in deltas if d["source"] == "only_live"]
    only_bt = [d for d in deltas if d["source"] == "only_backtest"]
    closed = [d for d in deltas if not d["is_open"]]
    still_open = [d for d in deltas if d["is_open"]]

    total = len(matched) + len(only_live) + len(only_bt)
    agreement = len(matched) / total if total else 0.0

    live_pnl = sum(d["live_pnl"] or 0 for d in closed)
    bt_pnl = sum(d["bt_pnl"] or 0 for d in closed)

    return {
        "trade_date": trade_date,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "n_live": len(matched) + len(only_live),
        "n_backtest": len(matched) + len(only_bt),
        "n_matched": len(matched),
        "n_only_live": len(only_live),
        "n_only_backtest": len(only_bt),
        "n_still_open": len(still_open),
        "agreement_rate": agreement,
        "live_total_pnl": live_pnl,
        "bt_total_pnl": bt_pnl,
        "pnl_delta": live_pnl - bt_pnl,
        "pnl_realization_ratio": live_pnl / bt_pnl if bt_pnl > 0 else None,
        "median_entry_slippage_bps": None,
        "median_exit_slippage_bps": None,
        "p95_entry_slippage_bps": None,
        "p95_exit_slippage_bps": None,
        "live_win_rate": None,
        "bt_win_rate": None,
        "win_rate_delta_pp": None,
        "equity_pnl_delta": sum(d["pnl_delta"] or 0 for d in matched
                                if d["instrument"] == "equity" and not d["is_open"]),
        "csp_pnl_delta": sum(d["pnl_delta"] or 0 for d in matched
                             if d["instrument"] == "csp" and not d["is_open"]),
        "tripwire_agreement_below_70pct": False,
        "tripwire_pnl_delta_negative_large": False,
        "tripwire_slippage_2x_assumption": False,
        "trade_deltas": deltas,
    }


def clear_history():
    h = TMPDIR / "history"
    if h.exists():
        shutil.rmtree(h)
    h.mkdir(parents=True, exist_ok=True)


def write_daily(report):
    path = TMPDIR / "history" / f"{report['trade_date']}.json"
    path.write_text(json.dumps(report, default=str))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_rollup():
    print("\n--- cumulative: basic 3-day rollup ---")
    clear_history()

    write_daily(make_daily_report("2026-05-21", [
        make_delta("AAPL", "2026-05-21", live_pnl=100.0, bt_pnl=80.0),
        make_delta("TSLA", "2026-05-21", live_pnl=-50.0, bt_pnl=-30.0),
    ]))
    write_daily(make_daily_report("2026-05-22", [
        make_delta("NVDA", "2026-05-22", live_pnl=200.0, bt_pnl=180.0),
    ]))
    write_daily(make_daily_report("2026-05-23", [
        make_delta("MSFT", "2026-05-23", live_pnl=150.0, bt_pnl=170.0),
    ]))

    daily = cumulative.load_all_daily_reports()
    report = cumulative.build_cumulative(daily)

    check("trading_days = 3", report.trading_days == 3)
    check("total_matched = 4", report.total_matched == 4)
    check("live_total_pnl = 100 - 50 + 200 + 150 = 400",
          report.live_total_pnl == 400.0, f"got {report.live_total_pnl}")
    check("bt_total_pnl = 80 - 30 + 180 + 170 = 400",
          report.bt_total_pnl == 400.0, f"got {report.bt_total_pnl}")
    check("pnl_delta_total = 0", report.pnl_delta_total == 0.0)
    check("realization = 100% when both are 400",
          report.pnl_realization_ratio == 1.0)


def test_negative_bt_pnl_total():
    print("\n--- cumulative: bt total is negative, ratio should be None ---")
    clear_history()

    # Across all days, backtest loses money but live loses less
    write_daily(make_daily_report("2026-05-21", [
        make_delta("AAPL", "2026-05-21", live_pnl=-50.0, bt_pnl=-200.0),
    ]))
    write_daily(make_daily_report("2026-05-22", [
        make_delta("TSLA", "2026-05-22", live_pnl=-100.0, bt_pnl=-300.0),
    ]))

    daily = cumulative.load_all_daily_reports()
    report = cumulative.build_cumulative(daily)

    check("pnl_delta_total = +350 (live lost less)",
          report.pnl_delta_total == 350.0, f"got {report.pnl_delta_total}")
    check("realization is None when bt <= 0",
          report.pnl_realization_ratio is None)


def test_dedup_on_rerun():
    print("\n--- cumulative: re-reconciled day shouldn't double-count ---")
    clear_history()

    # Day 1: 5/21 has an open CSP
    write_daily(make_daily_report("2026-05-21", [
        make_delta("AAPL", "2026-05-21", is_open=True, live_pnl=None, bt_pnl=None,
                   instrument="csp"),
    ], generated_at="2026-05-21T17:30:00+00:00"))

    # Day 2: 5/22 normal report
    write_daily(make_daily_report("2026-05-22", [
        make_delta("TSLA", "2026-05-22", live_pnl=100.0, bt_pnl=90.0),
    ], generated_at="2026-05-22T17:30:00+00:00"))

    # On 5/23 we re-reconcile 5/21 because the CSP closed.
    # The report for 5/21 gets overwritten with the closure data.
    write_daily(make_daily_report("2026-05-21", [
        make_delta("AAPL", "2026-05-21", is_open=False,
                   live_pnl=200.0, bt_pnl=180.0, instrument="csp"),
    ], generated_at="2026-05-23T17:30:00+00:00"))

    daily = cumulative.load_all_daily_reports()
    report = cumulative.build_cumulative(daily)

    # AAPL should count exactly once, with the closed values
    check("total_matched = 2 (AAPL + TSLA, not 3)",
          report.total_matched == 2, f"got {report.total_matched}")
    check("live_total_pnl = 200 + 100 = 300",
          report.live_total_pnl == 300.0, f"got {report.live_total_pnl}")
    check("CSP delta = 20 (counted once)",
          report.csp_pnl_delta_total == 20.0, f"got {report.csp_pnl_delta_total}")


def test_rolling_20_trips():
    print("\n--- cumulative: rolling-20 trips when win rate collapses ---")
    clear_history()

    # 30 trades: backtest wins 25, live wins only 5 → -67pp delta
    deltas = []
    for i in range(30):
        bt_pnl = 100.0 if i < 25 else -50.0
        live_pnl = 100.0 if i < 5 else -50.0
        td = f"2026-05-{(i % 5) + 1:02d}"  # spread across 5 days
        deltas.append(make_delta(f"TKR{i:02d}", td,
                                 live_pnl=live_pnl, bt_pnl=bt_pnl,
                                 live_entry_time=f"{td}T09:{35+i % 10:02d}:00+00:00"))

    # Group by trade_date
    by_date = {}
    for d in deltas:
        by_date.setdefault(d["trade_date"], []).append(d)
    for td, ds in by_date.items():
        write_daily(make_daily_report(td, ds))

    daily = cumulative.load_all_daily_reports()
    report = cumulative.build_cumulative(daily)

    check("rolling_win_rate_delta_pp computed",
          report.rolling_win_rate_delta_pp is not None)
    check("rolling_win_rate_delta_pp is large negative",
          report.rolling_win_rate_delta_pp < -15,
          f"got {report.rolling_win_rate_delta_pp}")
    check("tripwire_rolling_win_rate fires",
          report.tripwire_rolling_win_rate)


def test_rolling_20_underperform_with_neg_bt():
    print("\n--- cumulative: rolling-20 P&L tripwire with negative bt window ---")
    clear_history()

    # 20 trades. Backtest lost $1000. Live lost $1800 (live did much worse).
    deltas = []
    for i in range(20):
        bt_pnl = -50.0    # backtest: -1000 total
        live_pnl = -90.0  # live: -1800 total — underperform by $800 (>$500 = half of $1000)
        td = f"2026-05-{(i % 5) + 1:02d}"
        deltas.append(make_delta(f"TKR{i:02d}", td,
                                 live_pnl=live_pnl, bt_pnl=bt_pnl,
                                 live_entry_time=f"{td}T09:{35+i % 10:02d}:00+00:00"))

    by_date = {}
    for d in deltas:
        by_date.setdefault(d["trade_date"], []).append(d)
    for td, ds in by_date.items():
        write_daily(make_daily_report(td, ds))

    daily = cumulative.load_all_daily_reports()
    report = cumulative.build_cumulative(daily)

    check("rolling_pnl_delta = -$800 (live did $800 worse)",
          report.rolling_pnl_delta == -800.0,
          f"got {report.rolling_pnl_delta}")
    check("tripwire fires even with negative bt window",
          report.tripwire_rolling_pnl_underperform)


def test_rolling_20_no_trip_when_live_outperforms_neg_bt():
    print("\n--- cumulative: NO tripwire when live outperforms negative bt ---")
    clear_history()

    # 20 trades. Backtest lost $1000. Live lost only $500 (live did BETTER).
    deltas = []
    for i in range(20):
        bt_pnl = -50.0    # bt: -1000 total
        live_pnl = -25.0  # live: -500 total — BETTER than bt
        td = f"2026-05-{(i % 5) + 1:02d}"
        deltas.append(make_delta(f"TKR{i:02d}", td,
                                 live_pnl=live_pnl, bt_pnl=bt_pnl,
                                 live_entry_time=f"{td}T09:{35+i % 10:02d}:00+00:00"))

    by_date = {}
    for d in deltas:
        by_date.setdefault(d["trade_date"], []).append(d)
    for td, ds in by_date.items():
        write_daily(make_daily_report(td, ds))

    daily = cumulative.load_all_daily_reports()
    report = cumulative.build_cumulative(daily)

    check("rolling_pnl_delta is positive (live beat bt)",
          report.rolling_pnl_delta > 0,
          f"got {report.rolling_pnl_delta}")
    check("tripwire does NOT fire (live outperformed)",
          not report.tripwire_rolling_pnl_underperform)


def test_open_positions_excluded_from_pnl():
    print("\n--- cumulative: open positions don't pollute P&L stats ---")
    clear_history()

    write_daily(make_daily_report("2026-05-21", [
        make_delta("AAPL", "2026-05-21", live_pnl=100.0, bt_pnl=80.0),
        make_delta("OPENCSP", "2026-05-21", is_open=True, live_pnl=None, bt_pnl=None,
                   instrument="csp"),
    ]))

    daily = cumulative.load_all_daily_reports()
    report = cumulative.build_cumulative(daily)

    check("live_total_pnl = 100 (open excluded)",
          report.live_total_pnl == 100.0)
    check("total_matched = 2 (counts open trades)",
          report.total_matched == 2)
    check("total_closed_matched = 1",
          report.total_closed_matched == 1)


def test_empty_history():
    print("\n--- cumulative: no history at all ---")
    clear_history()

    daily = cumulative.load_all_daily_reports()
    check("load_all_daily_reports returns []", daily == [])

    try:
        cumulative.build_cumulative([])
        check("build_cumulative raises on empty", False, "did not raise")
    except ValueError:
        check("build_cumulative raises on empty", True)


def test_ks_drift_detection():
    print("\n--- cumulative: KS test detects score distribution drift ---")
    if not cumulative.HAVE_SCIPY:
        print("  (skipped — scipy not installed in test env)")
        return

    clear_history()

    # 30 matched trades where live scores cluster around 0.95 and bt around 0.60
    deltas = []
    for i in range(30):
        td = f"2026-05-{(i % 5) + 1:02d}"
        deltas.append(make_delta(f"TKR{i:02d}", td,
                                 live_pnl=10.0, bt_pnl=10.0,
                                 live_score=0.95, bt_score=0.60,
                                 live_entry_time=f"{td}T09:{35+i % 10:02d}:00+00:00"))

    by_date = {}
    for d in deltas:
        by_date.setdefault(d["trade_date"], []).append(d)
    for td, ds in by_date.items():
        write_daily(make_daily_report(td, ds))

    daily = cumulative.load_all_daily_reports()
    report = cumulative.build_cumulative(daily)

    check("ks_pvalue computed", report.ks_pvalue is not None)
    check("ks_pvalue is tiny (clear drift)", report.ks_pvalue < 0.01,
          f"got {report.ks_pvalue}")
    check("tripwire_score_distribution_drift fires",
          report.tripwire_score_distribution_drift)


def test_ks_no_drift_when_aligned():
    print("\n--- cumulative: KS does NOT trip when scores align ---")
    if not cumulative.HAVE_SCIPY:
        print("  (skipped — scipy not installed in test env)")
        return

    clear_history()

    # 30 trades where scores agree
    deltas = []
    for i in range(30):
        td = f"2026-05-{(i % 5) + 1:02d}"
        deltas.append(make_delta(f"TKR{i:02d}", td,
                                 live_pnl=10.0, bt_pnl=10.0,
                                 live_score=0.85 + (i * 0.001),  # tiny noise
                                 bt_score=0.85 + (i * 0.001),
                                 live_entry_time=f"{td}T09:{35+i % 10:02d}:00+00:00"))

    by_date = {}
    for d in deltas:
        by_date.setdefault(d["trade_date"], []).append(d)
    for td, ds in by_date.items():
        write_daily(make_daily_report(td, ds))

    daily = cumulative.load_all_daily_reports()
    report = cumulative.build_cumulative(daily)

    check("tripwire_score_distribution_drift does NOT fire",
          not report.tripwire_score_distribution_drift,
          f"ks_pvalue={report.ks_pvalue}")


def test_render_html_smoke():
    print("\n--- cumulative: render_html doesn't crash with edge values ---")
    clear_history()

    write_daily(make_daily_report("2026-05-21", [
        make_delta("AAPL", "2026-05-21", live_pnl=-50.0, bt_pnl=-200.0),
    ]))

    daily = cumulative.load_all_daily_reports()
    report = cumulative.build_cumulative(daily)
    try:
        html = cumulative.render_html(report)
        check("HTML produced", len(html) > 100)
        check("HTML contains 'n/a' for realization when bt<=0", "n/a" in html)
        check("HTML contains the date", "2026-05-21" in html)
    except Exception as e:
        check("HTML rendering didn't crash", False, str(e))


def run_all():
    test_basic_rollup()
    test_negative_bt_pnl_total()
    test_dedup_on_rerun()
    test_rolling_20_trips()
    test_rolling_20_underperform_with_neg_bt()
    test_rolling_20_no_trip_when_live_outperforms_neg_bt()
    test_open_positions_excluded_from_pnl()
    test_empty_history()
    test_ks_drift_detection()
    test_ks_no_drift_when_aligned()
    test_render_html_smoke()

    print(f"\n{'='*50}")
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"{PASS} ALL {total} CHECKS PASSED")
    else:
        print(f"{FAIL} {total - passed} of {total} checks FAILED")
        sys.exit(1)


if __name__ == "__main__":
    run_all()