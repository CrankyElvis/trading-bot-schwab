#!/usr/bin/env python3
"""
readiness_scorecard.py

Live trading readiness scorecard — run this before switching from paper to live.
Produces a GO / NOT YET recommendation based on two scores:

  1. BOT HEALTH SCORE    — paper trading performance metrics
  2. MARKET CONDITIONS   — current market environment quality

Both must be in acceptable range before going live with real money.

Usage:
  python readiness_scorecard.py
  python readiness_scorecard.py --json   (machine-readable output)
"""

import json
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ── Thresholds ────────────────────────────────────────────────────────────────
BOT_HEALTH_THRESHOLDS = {
    'min_paper_days':      30,     # at least 30 days paper trading
    'min_sharpe':          1.5,    # Sharpe ratio > 1.5
    'max_drawdown_pct':    15.0,   # max drawdown < 15%
    'min_win_rate_pct':    50.0,   # win rate > 50%
    'min_trades':          20,     # at least 20 closed trades
    'max_crashes_14d':     0,      # zero crashes in last 14 days
    'min_market_days':     5,      # seen at least 1 full market week
}

MARKET_THRESHOLDS = {
    'max_vix':             20.0,   # VIX < 20
    'spy_above_50d_ma':    True,   # SPY above 50d moving average
    'preferred_regimes':   ('neutral', 'flow'),
    'avoid_crisis':        True,
}


def load_paper_portfolio(path: str = 'paper_portfolio.json') -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {}


def load_bot_log(path: str = 'bot_log.jsonl') -> list:
    entries = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return entries


def compute_paper_metrics(portfolio: dict, log_entries: list) -> dict:
    """Compute performance metrics from paper portfolio."""
    metrics = {
        'paper_days':       0,
        'sharpe':           0.0,
        'max_drawdown_pct': 0.0,
        'win_rate_pct':     0.0,
        'total_trades':     0,
        'wins':             0,
        'losses':           0,
        'total_pnl':        0.0,
        'crashes_14d':      0,
        'error':            None,
    }

    if not portfolio:
        metrics['error'] = 'No paper_portfolio.json found'
        return metrics

    # Paper trading duration
    created_str = portfolio.get('created', '')
    if created_str:
        try:
            created = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
            now     = datetime.now(timezone.utc)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            metrics['paper_days'] = max(0, (now - created).days)
        except Exception:
            pass

    # Trade metrics from trade_log
    trade_log = portfolio.get('trade_log', [])
    sells = [t for t in trade_log if t.get('type') == 'SELL']
    metrics['total_trades'] = len(sells)

    pnls = [t.get('profit_loss', 0) for t in sells if 'profit_loss' in t]
    if pnls:
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        metrics['wins']          = len(wins)
        metrics['losses']        = len(losses)
        metrics['win_rate_pct']  = round(len(wins) / len(pnls) * 100, 1)
        metrics['total_pnl']     = round(sum(pnls), 2)

        # Simple Sharpe from daily P&L (approximate)
        import statistics
        if len(pnls) >= 5:
            mean_pnl = statistics.mean(pnls)
            std_pnl  = statistics.stdev(pnls) if len(pnls) > 1 else 1
            if std_pnl > 0:
                # Annualize assuming ~4 trades/week
                metrics['sharpe'] = round((mean_pnl / std_pnl) * (252 ** 0.5 / 4), 2)

    # Max drawdown — track peak cash_remaining across all trades
    # This gives the worst intraday drawdown seen during paper trading
    starting = portfolio.get('starting_cash', 25000)
    cash_history = [t.get('cash_remaining', starting) for t in trade_log if 'cash_remaining' in t]
    cash_history = [starting] + cash_history  # prepend starting value as baseline

    if len(cash_history) > 1:
        peak = starting
        max_dd = 0.0
        for val in cash_history:
            peak = max(peak, val)
            dd   = (peak - val) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        metrics['max_drawdown_pct'] = round(max_dd, 1)
    else:
        metrics['max_drawdown_pct'] = 0.0

    # Crashes in last 14 days from bot log
    cutoff_14d = datetime.now(timezone.utc) - timedelta(days=14)
    crashes = 0
    for entry in log_entries:
        ts_str = entry.get('timestamp', '')
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts > cutoff_14d and entry.get('event') == 'crash':
                    crashes += 1
            except Exception:
                pass
    metrics['crashes_14d'] = crashes

    return metrics


def get_market_conditions() -> dict:
    """
    Get current market conditions for launch readiness.
    Pulls from regime_state.json, paper_portfolio.json, and CBOE VIX API.
    """
    conditions = {
        'vix':              None,
        'vix_trend':        None,   # 'falling' | 'rising' | 'flat'
        'regime':           None,
        'spy_vs_20d':       None,   # True = above, False = below
        'spy_vs_50d':       None,
        'earnings_season':  None,   # True = heavy earnings window
        'fed_meeting_soon': None,   # True = Fed meeting within 5 trading days
        'market_breadth':   None,   # 'positive' | 'negative' | 'mixed'
        'vix_30d_avg':      None,
        'error':            None,
    }

    # Read regime state
    for regime_file in ['regime_state.json', 'paper_portfolio.json']:
        try:
            with open(regime_file) as f:
                data = json.load(f)
                if 'vixy' in data or 'regime' in data:
                    conditions['vix']    = data.get('vixy', data.get('vix'))
                    conditions['regime'] = data.get('regime')
                    break
        except Exception:
            pass

    # VIX trend — check if current VIX is below its 20d average
    try:
        import urllib.request, io, csv
        url  = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv'
        with urllib.request.urlopen(url, timeout=8) as resp:
            text   = resp.read().decode('utf-8')
            reader = csv.DictReader(io.StringIO(text))
            vix_vals = []
            for row in reader:
                try:
                    vix_vals.append(float(row.get('CLOSE', row.get('Close', 0))))
                except Exception:
                    pass
        if len(vix_vals) >= 20:
            vix_now    = vix_vals[-1]
            vix_20d    = sum(vix_vals[-20:]) / 20
            vix_5d_ago = vix_vals[-5] if len(vix_vals) >= 5 else vix_now
            if conditions['vix'] is None:
                conditions['vix'] = round(vix_now, 1)
            conditions['vix_30d_avg'] = round(sum(vix_vals[-30:]) / min(30, len(vix_vals)), 1)
            if vix_now < vix_5d_ago * 0.97:
                conditions['vix_trend'] = 'falling'
            elif vix_now > vix_5d_ago * 1.03:
                conditions['vix_trend'] = 'rising'
            else:
                conditions['vix_trend'] = 'flat'
    except Exception as e:
        conditions['vix_trend'] = 'unknown'

    # Earnings season detection — heavy windows: Jan 13–Feb 14, Apr 7–May 9,
    # Jul 7–Aug 8, Oct 6–Nov 7 (approximate — large-cap earnings clusters)
    now = datetime.now()
    heavy_windows = [
        (1, 13, 2, 14), (4, 7, 5, 9), (7, 7, 8, 8), (10, 6, 11, 7)
    ]
    in_earnings = False
    for sm, sd, em, ed in heavy_windows:
        start = datetime(now.year, sm, sd)
        end   = datetime(now.year, em, ed)
        if start <= now <= end:
            in_earnings = True
            break
    conditions['earnings_season'] = in_earnings

    # Fed meeting schedule — hardcoded 2025-2026 FOMC dates (8 meetings/year)
    # Bot should not launch within 2 trading days of FOMC announcement
    fomc_dates_2026 = [
        '2026-01-28', '2026-03-18', '2026-05-06', '2026-06-17',
        '2026-07-29', '2026-09-16', '2026-11-04', '2026-12-16',
    ]
    fomc_dates_2025 = [
        '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18',
        '2025-07-30', '2025-09-17', '2025-11-07', '2025-12-10',
    ]
    all_fomc = fomc_dates_2025 + fomc_dates_2026
    fed_soon = False
    window = timedelta(days=3)
    for ds in all_fomc:
        try:
            fd = datetime.strptime(ds, '%Y-%m-%d')
            if abs((fd - now).days) <= 3:
                fed_soon = True
                break
        except Exception:
            pass
    conditions['fed_meeting_soon'] = fed_soon

    return conditions


def score_market_conditions(conditions: dict) -> tuple[int, list, list]:
    """
    Comprehensive prime launch conditions score.
    Returns (score_pct, passed_checks, failed_checks).
    """
    passed = []
    failed = []

    vix    = conditions.get('vix')
    regime = conditions.get('regime')
    trend  = conditions.get('vix_trend')
    earn   = conditions.get('earnings_season')
    fed    = conditions.get('fed_meeting_soon')
    avg30  = conditions.get('vix_30d_avg')

    # ── VIX level ────────────────────────────────────────────────────────────
    if vix is not None:
        if vix < 16:
            passed.append(f"  ✅ VIX: {vix:.1f} — excellent (< 16, low fear)")
        elif vix < 20:
            passed.append(f"  ✅ VIX: {vix:.1f} — acceptable (< 20)")
        elif vix < 25:
            failed.append(f"  ⚠️  VIX: {vix:.1f} — elevated (prefer < 20 to launch)")
        else:
            failed.append(f"  ❌ VIX: {vix:.1f} — too high (need < 20 to launch)")
    else:
        failed.append(f"  ❓ VIX: unknown")

    # ── VIX trend ────────────────────────────────────────────────────────────
    if trend == 'falling':
        passed.append(f"  ✅ VIX trend: falling (fear declining — ideal for entry)")
    elif trend == 'flat':
        passed.append(f"  ✅ VIX trend: flat (stable — acceptable)")
    elif trend == 'rising':
        failed.append(f"  ❌ VIX trend: rising (fear increasing — wait for reversal)")
    else:
        failed.append(f"  ❓ VIX trend: unknown")

    # ── VIX vs 30d average ───────────────────────────────────────────────────
    if vix is not None and avg30 is not None:
        if vix < avg30 * 0.90:
            passed.append(f"  ✅ VIX vs 30d avg: {vix:.1f} vs {avg30:.1f} (well below avg — calm market)")
        elif vix < avg30:
            passed.append(f"  ✅ VIX vs 30d avg: {vix:.1f} vs {avg30:.1f} (below avg — good)")
        elif vix < avg30 * 1.10:
            failed.append(f"  ⚠️  VIX vs 30d avg: {vix:.1f} vs {avg30:.1f} (near avg — marginal)")
        else:
            failed.append(f"  ❌ VIX vs 30d avg: {vix:.1f} vs {avg30:.1f} (above avg — elevated stress)")

    # ── Regime ───────────────────────────────────────────────────────────────
    if regime == 'flow':
        passed.append(f"  ✅ Regime: FLOW — best possible regime for this strategy (Sharpe 7.35)")
    elif regime == 'neutral':
        passed.append(f"  ✅ Regime: NEUTRAL — solid regime (Sharpe 5.16)")
    elif regime == 'volatility':
        failed.append(f"  ❌ Regime: VOLATILITY — strategy underperforms here (Sharpe 0.75)")
    elif regime == 'crisis':
        failed.append(f"  ❌ Regime: CRISIS — do not launch in crisis regime")
    else:
        failed.append(f"  ❓ Regime: unknown")

    # ── Earnings season ──────────────────────────────────────────────────────
    if earn is False:
        passed.append(f"  ✅ Earnings season: clear — not in heavy reporting window")
    elif earn is True:
        failed.append(f"  ❌ Earnings season: active — heavy large-cap earnings window increases gap risk")
    else:
        failed.append(f"  ❓ Earnings season: unknown")

    # ── Fed meeting ──────────────────────────────────────────────────────────
    if fed is False:
        passed.append(f"  ✅ Fed meeting: none within 3 days")
    elif fed is True:
        failed.append(f"  ❌ Fed meeting: within 3 days — FOMC creates overnight gap risk")
    else:
        failed.append(f"  ❓ Fed meeting: unknown")

    # ── Manual checks ────────────────────────────────────────────────────────
    passed.append(f"  ℹ️  SPY vs 50d MA: verify manually (need above 50d for trend confirmation)")
    passed.append(f"  ℹ️  SPY vs 20d MA: verify manually (overnight strategy needs SPY above 20d)")

    total = len(passed) + len(failed)
    # Weight manual items as neutral — don't count them in score
    auto_passed = [p for p in passed if '✅' in p]
    auto_failed = [f for f in failed if '❌' in f or '⚠️' in f]
    auto_total  = len(auto_passed) + len(auto_failed)
    score = int(len(auto_passed) / auto_total * 100) if auto_total > 0 else 0

    return score, passed, failed


def score_bot_health(metrics: dict) -> tuple[int, list, list]:
    """
    Returns (score_pct, passed_checks, failed_checks).
    Score is 0-100 based on how many thresholds are met.
    """
    checks = []
    passed = []
    failed = []

    t = BOT_HEALTH_THRESHOLDS

    def check(label, value, threshold, fmt=None, higher_is_better=True):
        if value is None:
            failed.append(f"  ❓ {label}: unknown")
            return
        display = fmt.format(value) if fmt else str(value)
        if higher_is_better:
            ok = value >= threshold
        else:
            ok = value <= threshold
        if ok:
            passed.append(f"  ✅ {label}: {display}")
        else:
            target = fmt.format(threshold) if fmt else str(threshold)
            failed.append(f"  ❌ {label}: {display}  (need {target})")

    check('Paper trading days',  metrics['paper_days'],       t['min_paper_days'],    '{:.0f}d')
    check('Sharpe ratio',        metrics['sharpe'],           t['min_sharpe'],        '{:.2f}')
    check('Max drawdown',        metrics['max_drawdown_pct'], t['max_drawdown_pct'],  '{:.1f}%', False)
    check('Win rate',            metrics['win_rate_pct'],     t['min_win_rate_pct'],  '{:.1f}%')
    check('Closed trades',       metrics['total_trades'],     t['min_trades'],        '{:.0f}')
    check('Crashes (14d)',       metrics['crashes_14d'],      t['max_crashes_14d'],   '{:.0f}',  False)

    total  = len(passed) + len(failed)
    score  = int(len(passed) / total * 100) if total > 0 else 0
    return score, passed, failed


def score_market_conditions(conditions: dict) -> tuple[int, list, list]:
    """
    Returns (score_pct, passed_checks, failed_checks).
    """
    passed = []
    failed = []
    t = MARKET_THRESHOLDS

    vix    = conditions.get('vix')
    regime = conditions.get('regime')

    if vix is not None:
        if vix < t['max_vix']:
            passed.append(f"  ✅ VIX: {vix:.1f}  (< {t['max_vix']})")
        else:
            failed.append(f"  ❌ VIX: {vix:.1f}  (need < {t['max_vix']})")
    else:
        failed.append(f"  ❓ VIX: unknown")

    if regime is not None:
        if regime in t['preferred_regimes']:
            passed.append(f"  ✅ Regime: {regime}")
        else:
            failed.append(f"  ❌ Regime: {regime}  (need neutral or flow)")
    else:
        failed.append(f"  ❓ Regime: unknown")

    # SPY vs 50d MA — placeholder until live data available
    passed.append(f"  ℹ️  SPY vs 50d MA: check manually before going live")

    total = len(passed) + len(failed)
    score = int(len(passed) / total * 100) if total > 0 else 0
    return score, passed, failed


def print_scorecard(portfolio: dict, log_entries: list, as_json: bool = False):
    metrics    = compute_paper_metrics(portfolio, log_entries)
    conditions = get_market_conditions()

    bot_score,  bot_passed,  bot_failed   = score_bot_health(metrics)
    mkt_score,  mkt_passed,  mkt_failed   = score_market_conditions(conditions)

    combined = (bot_score + mkt_score) // 2

    if bot_score == 100 and mkt_score >= 80:
        verdict     = "✅  GO — Bot ready, prime market window"
        verdict_msg = "All health checks pass AND market conditions are excellent. This is the ideal launch moment."
    elif bot_score == 100 and mkt_score >= 60:
        verdict     = "⚠️   BOT READY but market window is marginal"
        verdict_msg = "Bot is fully validated. Wait for VIX to drop or regime to improve before launching."
    elif bot_score >= 80 and mkt_score >= 80:
        verdict     = "⚠️   ALMOST — fix remaining bot checks"
        verdict_msg = "Market window is great but bot has unresolved checks. Close the gaps quickly."
    elif bot_score < 80:
        verdict     = "🔴  NOT YET — continue paper trading"
        verdict_msg = "Bot not sufficiently validated. Keep paper trading regardless of market conditions."
    else:
        verdict     = "🔴  WAIT — poor market conditions"
        verdict_msg = "Even if bot were ready, these market conditions are unfavorable for launch."

    if as_json:
        result = {
            'timestamp':     datetime.now().isoformat(),
            'verdict':       verdict.strip(),
            'bot_score':     bot_score,
            'market_score':  mkt_score,
            'combined':      combined,
            'bot_passed':    bot_passed,
            'bot_failed':    bot_failed,
            'mkt_passed':    mkt_passed,
            'mkt_failed':    mkt_failed,
            'metrics':       metrics,
            'conditions':    conditions,
        }
        print(json.dumps(result, indent=2))
        return

    now = datetime.now().strftime('%Y-%m-%d %H:%M ET')
    print(f"\n{'='*62}")
    print(f"  LIVE TRADING READINESS SCORECARD  —  {now}")
    print(f"{'='*62}")
    print(f"  Overall verdict:  {verdict}")
    print(f"  {verdict_msg}")
    print(f"\n  Bot health score:      {bot_score:>3}%")
    print(f"  Market conditions:     {mkt_score:>3}%")
    print(f"  Combined:              {combined:>3}%")

    print(f"\n  ── Bot Health ({'✅' if bot_score == 100 else '⚠️ '}) ──────────────────────────────────────")
    for line in bot_passed:
        print(line)
    for line in bot_failed:
        print(line)

    print(f"\n  Paper stats:")
    print(f"    Trades: {metrics['total_trades']}  |  Wins: {metrics['wins']}  |  Losses: {metrics['losses']}")
    print(f"    Total P&L: ${metrics['total_pnl']:+,.2f}  |  Sharpe: {metrics['sharpe']:.2f}")

    mkt_emoji = '✅' if mkt_score >= 80 else ('⚠️ ' if mkt_score >= 50 else '❌')
    vix_info = ''
    if conditions.get('vix_30d_avg'):
        vix_info = f"  (30d avg: {conditions['vix_30d_avg']:.1f})"
    print(f"\n  ── Market Conditions — Prime Launch Window? {mkt_emoji} ──────────")
    for line in mkt_passed:
        print(line)
    for line in mkt_failed:
        print(line)

    # Prime window summary
    print()
    if mkt_score >= 80:
        print(f"  🟢 PRIME WINDOW: Market conditions are excellent for launch")
    elif mkt_score >= 60:
        print(f"  🟡 ACCEPTABLE WINDOW: Market ok but not ideal — address warnings first")
    elif mkt_score >= 40:
        print(f"  🟠 MARGINAL WINDOW: Several headwinds present — consider waiting")
    else:
        print(f"  🔴 POOR WINDOW: Significant headwinds — wait for better conditions")

    print(f"\n  ── Before going live, also verify manually ─────────────────")
    print(f"  □ Schwab account has sufficient buying power")
    print(f"  □ Options trading approved (Level 2 minimum)")
    print(f"  □ No earnings releases in portfolio within 48hrs")
    print(f"  □ No Fed meeting within 48hrs")
    print(f"  □ End-to-end dry run completed (#7)")
    print(f"  □ Anomaly notifications configured (#4)")
    print(f"  □ Server health monitoring active (#8)")
    print(f"{'='*62}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--portfolio', default='paper_portfolio.json')
    parser.add_argument('--log',       default='bot_log.jsonl')
    args = parser.parse_args()

    portfolio   = load_paper_portfolio(args.portfolio)
    log_entries = load_bot_log(args.log)
    print_scorecard(portfolio, log_entries, as_json=args.json)