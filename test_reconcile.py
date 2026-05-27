"""
Test harness for v2 reconcile.py.

Stubs out the bot.storage and backtester.engine imports with fakes so we can
exercise the matching, helper, and aggregation logic against controlled data.

Run: python test_reconcile.py
"""

import sys
import types
import json
from datetime import datetime, timedelta, date, timezone
from pathlib import Path
import tempfile

import pandas as pd


# ---------------------------------------------------------------------------
# Stub modules BEFORE importing reconcile
# ---------------------------------------------------------------------------

bot_mod = types.ModuleType("bot")
bot_storage = types.ModuleType("bot.storage")
bot_config = types.ModuleType("bot.config")
bot_email = types.ModuleType("bot.email_alerts")
bt_mod = types.ModuleType("backtester")
bt_engine = types.ModuleType("backtester.engine")

# Configure VALIDATION_DIR to a temp directory
TMPDIR = Path(tempfile.mkdtemp())
bot_config.VALIDATION_DIR = str(TMPDIR)
bot_config.ALERT_EMAIL = "test@example.com"

# Stub load_paper_trades — set by each test
_paper_trades_fixture = pd.DataFrame()
def _load_paper_trades(entry_date_start=None, entry_date_end=None, **kw):
    return _paper_trades_fixture.copy()
bot_storage.load_paper_trades = _load_paper_trades

# Stub run_backtest
_bt_trades_fixture = pd.DataFrame()
class _BTResult:
    def __init__(self, trade_log):
        self.trade_log = trade_log
def _run_backtest(start_date, end_date, data_snapshot, **kw):
    return _BTResult(_bt_trades_fixture.copy())
bt_engine.run_backtest = _run_backtest

bot_email.send_email = lambda **kw: None

sys.modules["bot"] = bot_mod
sys.modules["bot.storage"] = bot_storage
sys.modules["bot.config"] = bot_config
sys.modules["bot.email_alerts"] = bot_email
sys.modules["backtester"] = bt_mod
sys.modules["backtester.engine"] = bt_engine

# Create snapshots dir + a fake snapshot file so load_backtest_trades doesn't raise
(TMPDIR / "snapshots").mkdir(parents=True, exist_ok=True)


# Now safe to import
sys.path.insert(0, str(Path(__file__).parent))
import reconcile  # noqa: E402


# ---------------------------------------------------------------------------
# Test utilities
# ---------------------------------------------------------------------------

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


def make_trade(ticker, entry_time, entry_price, exit_price=None, pnl=None,
               exit_time=None, exit_reason=None, score=0.85, regime="neutral",
               signals=None, instrument="equity"):
    """Construct a single trade dict."""
    return {
        "ticker": ticker,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "pnl": pnl,
        "exit_reason": exit_reason,
        "score": score,
        "regime": regime,
        "signals_fired": signals or ["momentum"],
        "instrument": instrument,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_safe_helpers():
    print("\n--- _safe_iso / _safe_float / _safe_str across types ---")

    # _safe_iso
    check("_safe_iso(None) -> None", reconcile._safe_iso(None) is None)
    check("_safe_iso('2026-05-23T09:35:00') -> string passthrough",
          reconcile._safe_iso("2026-05-23T09:35:00") == "2026-05-23T09:35:00")
    dt = datetime(2026, 5, 23, 9, 35)
    check("_safe_iso(datetime) -> isoformat",
          reconcile._safe_iso(dt) == "2026-05-23T09:35:00")
    ts = pd.Timestamp("2026-05-23T09:35:00", tz="UTC")
    check("_safe_iso(Timestamp) -> isoformat",
          reconcile._safe_iso(ts) is not None and "2026-05-23" in reconcile._safe_iso(ts))
    check("_safe_iso(NaT) -> None", reconcile._safe_iso(pd.NaT) is None)
    check("_safe_iso(NaN) -> None", reconcile._safe_iso(float("nan")) is None)

    # _safe_float
    check("_safe_float(None) -> None", reconcile._safe_float(None) is None)
    check("_safe_float(NaN) -> None", reconcile._safe_float(float("nan")) is None)
    check("_safe_float(NaT) -> None", reconcile._safe_float(pd.NaT) is None)
    check("_safe_float('3.14') -> 3.14", reconcile._safe_float("3.14") == 3.14)
    check("_safe_float('garbage') -> None", reconcile._safe_float("garbage") is None)
    check("_safe_float(0) -> 0.0", reconcile._safe_float(0) == 0.0)

    # _bps
    check("_bps with 0 denominator -> None", reconcile._bps(100, 0) is None)
    check("_bps(101, 100) -> 100", reconcile._bps(101, 100) == 100.0)
    check("_bps with None -> None", reconcile._bps(None, 100) is None)
    check("_bps with string inputs", reconcile._bps("101", "100") == 100.0)


def test_match_exact():
    print("\n--- match_trades: clean 1:1 match ---")
    t = pd.Timestamp("2026-05-23T09:35:00", tz="UTC")
    live = pd.DataFrame([
        make_trade("AAPL", t, 200.0, 205.0, 500.0, t + timedelta(hours=2), "profit_target"),
        make_trade("TSLA", t + timedelta(minutes=5), 300.0, 295.0, -500.0,
                   t + timedelta(hours=3), "stop"),
    ])
    bt = pd.DataFrame([
        make_trade("AAPL", t + timedelta(seconds=30), 199.5, 204.0, 450.0,
                   t + timedelta(hours=2), "profit_target"),
        make_trade("TSLA", t + timedelta(minutes=5, seconds=10), 300.5, 296.0, -450.0,
                   t + timedelta(hours=3), "stop"),
    ])

    deltas = reconcile.match_trades(live, bt)
    check("2 deltas produced", len(deltas) == 2, f"got {len(deltas)}")
    check("both matched", all(d.matched for d in deltas))
    check("both sourced 'both'", all(d.source == "both" for d in deltas))
    aapl = next(d for d in deltas if d.ticker == "AAPL")
    check("AAPL entry slip bps positive (live paid more)",
          aapl.entry_slippage_bps is not None and aapl.entry_slippage_bps > 0,
          f"got {aapl.entry_slippage_bps}")
    check("AAPL pnl_delta = 500 - 450 = 50",
          aapl.pnl_delta == 50.0, f"got {aapl.pnl_delta}")


def test_match_only_live():
    print("\n--- match_trades: live trade with no bt counterpart ---")
    t = pd.Timestamp("2026-05-23T09:35:00", tz="UTC")
    live = pd.DataFrame([make_trade("AAPL", t, 200.0, 205.0, 500.0, t + timedelta(hours=2))])
    bt = pd.DataFrame()
    deltas = reconcile.match_trades(live, bt)
    check("1 delta", len(deltas) == 1)
    check("source = only_live", deltas[0].source == "only_live")
    check("not matched", not deltas[0].matched)


def test_match_only_bt():
    print("\n--- match_trades: bt trade with no live counterpart ---")
    t = pd.Timestamp("2026-05-23T09:35:00", tz="UTC")
    live = pd.DataFrame()
    bt = pd.DataFrame([make_trade("AAPL", t, 200.0, 205.0, 500.0, t + timedelta(hours=2))])
    deltas = reconcile.match_trades(live, bt)
    check("1 delta", len(deltas) == 1)
    check("source = only_backtest", deltas[0].source == "only_backtest")


def test_match_outside_tolerance():
    print("\n--- match_trades: same ticker but >120s entry gap ---")
    t = pd.Timestamp("2026-05-23T09:35:00", tz="UTC")
    live = pd.DataFrame([make_trade("AAPL", t, 200.0)])
    # 200 seconds = outside 120s tolerance
    bt = pd.DataFrame([make_trade("AAPL", t + timedelta(seconds=200), 200.0)])
    deltas = reconcile.match_trades(live, bt)
    check("2 deltas (one each side)", len(deltas) == 2)
    sources = sorted(d.source for d in deltas)
    check("one only_live, one only_backtest",
          sources == ["only_backtest", "only_live"], f"got {sources}")


def test_match_double_entry_same_day():
    print("\n--- match_trades: same ticker entered twice, must use each bt once ---")
    t = pd.Timestamp("2026-05-23T09:35:00", tz="UTC")
    live = pd.DataFrame([
        make_trade("AAPL", t, 200.0),
        make_trade("AAPL", t + timedelta(hours=3), 201.0),
    ])
    bt = pd.DataFrame([
        make_trade("AAPL", t + timedelta(seconds=10), 200.0),
        make_trade("AAPL", t + timedelta(hours=3, seconds=10), 201.0),
    ])
    deltas = reconcile.match_trades(live, bt)
    check("2 deltas", len(deltas) == 2)
    check("both matched", all(d.matched for d in deltas))


def test_build_report_negative_bt_pnl():
    print("\n--- build_report: negative bt P&L doesn't flip realization sign ---")
    t = pd.Timestamp("2026-05-23T09:35:00", tz="UTC")
    # Backtest lost $200, live lost $50 — live did BETTER
    live = pd.DataFrame([make_trade("AAPL", t, 200.0, 199.5, -50.0,
                                    t + timedelta(hours=2), "stop")])
    bt = pd.DataFrame([make_trade("AAPL", t + timedelta(seconds=10), 200.0, 198.0,
                                  -200.0, t + timedelta(hours=2), "stop")])

    deltas = reconcile.match_trades(live, bt)
    report = reconcile.build_report(date(2026, 5, 23), deltas)

    check("pnl_delta is +150 (live beat bt by $150)",
          report.pnl_delta == 150.0, f"got {report.pnl_delta}")
    check("pnl_realization_ratio is None when bt <= 0",
          report.pnl_realization_ratio is None,
          f"got {report.pnl_realization_ratio}")
    check("tripwire_pnl_delta_negative_large is False (live OUTperformed)",
          not report.tripwire_pnl_delta_negative_large)


def test_build_report_open_position():
    print("\n--- build_report: open position excluded from P&L stats ---")
    t = pd.Timestamp("2026-05-23T09:35:00", tz="UTC")
    # CSP still open — no exit data
    live = pd.DataFrame([make_trade("AAPL", t, 5.0, exit_price=None, pnl=None,
                                    instrument="csp")])
    bt = pd.DataFrame([make_trade("AAPL", t + timedelta(seconds=10), 5.0,
                                  exit_price=None, pnl=None, instrument="csp")])
    deltas = reconcile.match_trades(live, bt)
    report = reconcile.build_report(date(2026, 5, 23), deltas)

    check("n_matched = 1", report.n_matched == 1)
    check("n_still_open = 1", report.n_still_open == 1)
    check("live_total_pnl = 0 (open trades skipped)", report.live_total_pnl == 0.0)
    check("bt_total_pnl = 0 (open trades skipped)", report.bt_total_pnl == 0.0)
    check("pnl_delta = 0 (open trades skipped)", report.pnl_delta == 0.0)


def test_build_report_string_timestamps():
    print("\n--- build_report: string entry_time (SQLite TEXT) doesn't crash ---")
    # Some storage layers return ISO strings, not Timestamps
    live = pd.DataFrame([{
        "ticker": "AAPL",
        "entry_time": pd.Timestamp("2026-05-23T09:35:00", tz="UTC"),
        "entry_price": 200.0,
        "exit_time": "2026-05-23T11:35:00",  # string!
        "exit_price": 205.0,
        "pnl": 500.0,
        "exit_reason": "profit_target",
        "score": 0.85, "regime": "neutral", "signals_fired": ["momentum"],
        "instrument": "equity",
    }])
    bt = pd.DataFrame([{
        "ticker": "AAPL",
        "entry_time": pd.Timestamp("2026-05-23T09:35:10", tz="UTC"),
        "entry_price": 199.5,
        "exit_time": pd.Timestamp("2026-05-23T11:35:00", tz="UTC"),
        "exit_price": 204.0,
        "pnl": 450.0,
        "exit_reason": "profit_target",
        "score": 0.85, "regime": "neutral", "signals_fired": ["momentum"],
        "instrument": "equity",
    }])

    try:
        deltas = reconcile.match_trades(live, bt)
        report = reconcile.build_report(date(2026, 5, 23), deltas)
        check("no crash on string exit_time", True)
        check("trade matched", report.n_matched == 1)
        check("pnl_delta correct", report.pnl_delta == 50.0)
        # The persist round-trip is important — make sure JSON-serializable
        path = reconcile.persist_report(report)
        loaded = json.loads(path.read_text())
        check("report round-trips through JSON",
              loaded["trade_date"] == "2026-05-23")
    except Exception as e:
        check("no crash on string exit_time", False, f"raised {type(e).__name__}: {e}")


def test_build_report_empty_day():
    print("\n--- build_report: no trades either side ---")
    report = reconcile.build_report(date(2026, 5, 23), [])
    check("n_live = 0", report.n_live == 0)
    check("n_backtest = 0", report.n_backtest == 0)
    check("agreement_rate = 0", report.agreement_rate == 0.0)
    check("pnl_realization_ratio is None", report.pnl_realization_ratio is None)
    check("no tripwires", not report.tripwire_agreement_below_70pct
          and not report.tripwire_pnl_delta_negative_large)


def test_build_report_agreement_tripwire():
    print("\n--- build_report: agreement tripwire requires >=5 trades ---")
    t = pd.Timestamp("2026-05-23T09:35:00", tz="UTC")
    # 3 only_live, 0 matched, 0 only_bt — agreement = 0 but n < 5
    live_rows = [make_trade(f"TKR{i}", t + timedelta(minutes=i), 100.0,
                            105.0, 100.0, t + timedelta(hours=2))
                 for i in range(3)]
    live = pd.DataFrame(live_rows)
    bt = pd.DataFrame()
    deltas = reconcile.match_trades(live, bt)
    report = reconcile.build_report(date(2026, 5, 23), deltas)
    check("agreement = 0 with 3 trades",
          report.agreement_rate == 0.0)
    check("agreement tripwire NOT fired (n < 5)",
          not report.tripwire_agreement_below_70pct)

    # Now 6 only_live
    live = pd.DataFrame([make_trade(f"TKR{i}", t + timedelta(minutes=i), 100.0,
                                    105.0, 100.0, t + timedelta(hours=2))
                         for i in range(6)])
    deltas = reconcile.match_trades(live, bt)
    report = reconcile.build_report(date(2026, 5, 23), deltas)
    check("agreement tripwire FIRES (n >= 5)",
          report.tripwire_agreement_below_70pct)


def test_match_index_safety():
    print("\n--- match_trades: non-default pandas index doesn't break matching ---")
    t = pd.Timestamp("2026-05-23T09:35:00", tz="UTC")
    live = pd.DataFrame([make_trade("AAPL", t, 200.0, 205.0, 500.0,
                                    t + timedelta(hours=2))])
    bt = pd.DataFrame([make_trade("AAPL", t + timedelta(seconds=10), 199.5, 204.0,
                                  450.0, t + timedelta(hours=2))],
                      index=[99])  # weird index
    deltas = reconcile.match_trades(live, bt)
    check("1 matched delta despite weird bt index",
          len(deltas) == 1 and deltas[0].matched)


def run_all():
    test_safe_helpers()
    test_match_exact()
    test_match_only_live()
    test_match_only_bt()
    test_match_outside_tolerance()
    test_match_double_entry_same_day()
    test_match_index_safety()
    test_build_report_negative_bt_pnl()
    test_build_report_open_position()
    test_build_report_string_timestamps()
    test_build_report_empty_day()
    test_build_report_agreement_tripwire()

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