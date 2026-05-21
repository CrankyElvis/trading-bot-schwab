# -*- coding: utf-8 -*-
"""
readiness_scorecard.py  —  Live trading readiness scorecard

Run this before switching from paper to live trading.
Produces a GO / NOT YET recommendation based on:
  1. BOT HEALTH SCORE    -- paper trading performance metrics
  2. MARKET CONDITIONS   -- current market environment (prime window check)

Usage:
  python readiness_scorecard.py
  python readiness_scorecard.py --json
"""

import json
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ── Thresholds ────────────────────────────────────────────────────────────────
BOT_HEALTH_THRESHOLDS = {
    'min_paper_days':      30,
    'min_sharpe':          1.5,
    'max_drawdown_pct':    15.0,
    'min_win_rate_pct':    50.0,
    'min_trades':          20,
    'max_crashes_14d':     0,
}

# FOMC meeting dates 2025-2026
FOMC_DATES = [
    '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18',
    '2025-07-30', '2025-09-17', '2025-11-07', '2025-12-10',
    '2026-01-28', '2026-03-18', '2026-05-06', '2026-06-17',
    '2026-07-29', '2026-09-16', '2026-11-04', '2026-12-16',
]

# Heavy earnings windows (approximate large-cap clusters)
EARNINGS_WINDOWS = [
    (1, 13, 2, 14), (4, 7, 5, 9), (7, 7, 8, 8), (10, 6, 11, 7)
]


def load_paper_portfolio(path='paper_portfolio.json') -> dict:
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def load_bot_log(path='bot_log.jsonl') -> list:
    entries = []
    try:
        with open(path, encoding='utf-8') as f:
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
    metrics = {
        'paper_days': 0, 'sharpe': 0.0, 'max_drawdown_pct': 0.0,
        'win_rate_pct': 0.0, 'total_trades': 0, 'wins': 0,
        'losses': 0, 'total_pnl': 0.0, 'crashes_14d': 0,
    }
    if not portfolio:
        return metrics

    created_str = portfolio.get('created', '')
    if created_str:
        try:
            created = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            metrics['paper_days'] = max(0, (now - created).days)
        except Exception:
            pass

    trade_log = portfolio.get('trade_log', [])
    sells = [t for t in trade_log if t.get('type') == 'SELL']
    metrics['total_trades'] = len(sells)

    pnls = [t.get('profit_loss', 0) for t in sells if 'profit_loss' in t]
    if pnls:
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        metrics['wins']         = len(wins)
        metrics['losses']       = len(losses)
        metrics['win_rate_pct'] = round(len(wins) / len(pnls) * 100, 1)
        metrics['total_pnl']    = round(sum(pnls), 2)
        import statistics
        if len(pnls) >= 5:
            std = statistics.stdev(pnls) if len(pnls) > 1 else 1
            if std > 0:
                metrics['sharpe'] = round((statistics.mean(pnls) / std) * (252 ** 0.5 / 4), 2)

    starting = portfolio.get('starting_cash', 25000)
    cash_history = [t.get('cash_remaining', starting) for t in trade_log if 'cash_remaining' in t]
    cash_history = [starting] + cash_history
    if len(cash_history) > 1:
        peak, max_dd = starting, 0.0
        for val in cash_history:
            peak  = max(peak, val)
            dd    = (peak - val) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        metrics['max_drawdown_pct'] = round(max_dd, 1)

    cutoff_14d = datetime.now(timezone.utc) - timedelta(days=14)
    for entry in log_entries:
        ts_str = entry.get('timestamp', '')
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts > cutoff_14d and entry.get('event') == 'error':
                    metrics['crashes_14d'] += 1
            except Exception:
                pass

    return metrics


def get_market_conditions() -> dict:
    conditions = {
        'vix': None, 'vix_trend': None, 'regime': None,
        'earnings_season': None, 'fed_meeting_soon': None,
        'vix_30d_avg': None,
    }

    # Read regime and vixy from bot_log.jsonl last cycle entry
    try:
        with open('bot_log.jsonl', encoding='utf-8') as f:
            lines_raw = [l.strip() for l in f if l.strip()]
        for raw in reversed(lines_raw):
            try:
                entry = json.loads(raw)
                if entry.get('regime'):
                    conditions['regime'] = entry.get('regime')
                    conditions['vix']    = entry.get('vixy', entry.get('vix'))
                    break
            except Exception:
                pass
    except Exception:
        pass
    # Fallback: try paper_portfolio.json
    if not conditions['regime']:
        try:
            with open('paper_portfolio.json', encoding='utf-8') as f:
                data = json.load(f)
            conditions['vix']    = data.get('vixy', data.get('vix'))
            conditions['regime'] = data.get('regime')
        except Exception:
            pass

    try:
        import urllib.request, io, csv
        url = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv'
        with urllib.request.urlopen(url, timeout=8) as resp:
            text = resp.read().decode('utf-8')
        reader = csv.DictReader(io.StringIO(text))
        vix_vals = []
        for row in reader:
            try:
                vix_vals.append(float(row.get('CLOSE', row.get('Close', 0))))
            except Exception:
                pass
        if len(vix_vals) >= 20:
            vix_now = vix_vals[-1]
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
    except Exception:
        conditions['vix_trend'] = 'unknown'

    now = datetime.now()
    in_earnings = any(
        datetime(now.year, sm, sd) <= now <= datetime(now.year, em, ed)
        for sm, sd, em, ed in EARNINGS_WINDOWS
    )
    conditions['earnings_season'] = in_earnings

    fed_soon = any(
        abs((datetime.strptime(ds, '%Y-%m-%d') - now).days) <= 3
        for ds in FOMC_DATES
    )
    conditions['fed_meeting_soon'] = fed_soon

    return conditions


def score_bot_health(metrics: dict) -> tuple:
    t = BOT_HEALTH_THRESHOLDS
    passed, failed = [], []

    def chk(label, val, threshold, fmt='{}', hi=True):
        display = fmt.format(val) if val is not None else 'unknown'
        ok = (val >= threshold) if hi else (val <= threshold)
        if ok:
            passed.append(f'  [OK] {label}: {display}')
        else:
            target = fmt.format(threshold)
            failed.append(f'  [NO] {label}: {display}  (need {target})')

    chk('Paper trading days',  metrics['paper_days'],       t['min_paper_days'],    '{:.0f}d')
    chk('Sharpe ratio',        metrics['sharpe'],           t['min_sharpe'],        '{:.2f}')
    chk('Max drawdown',        metrics['max_drawdown_pct'], t['max_drawdown_pct'],  '{:.1f}%', False)
    chk('Win rate',            metrics['win_rate_pct'],     t['min_win_rate_pct'],  '{:.1f}%')
    chk('Closed trades',       metrics['total_trades'],     t['min_trades'],        '{:.0f}')
    chk('Crashes (14d)',       metrics['crashes_14d'],      t['max_crashes_14d'],   '{:.0f}',  False)

    total = len(passed) + len(failed)
    score = int(len(passed) / total * 100) if total > 0 else 0
    return score, passed, failed


def score_market_conditions(conditions: dict) -> tuple:
    passed, failed = [], []

    vix    = conditions.get('vix')
    regime = conditions.get('regime')
    trend  = conditions.get('vix_trend')
    earn   = conditions.get('earnings_season')
    fed    = conditions.get('fed_meeting_soon')
    avg30  = conditions.get('vix_30d_avg')

    # VIX level
    if vix is not None:
        if vix < 16:
            passed.append(f'  [OK] VIX: {vix:.1f} -- excellent (< 16, low fear)')
        elif vix < 20:
            passed.append(f'  [OK] VIX: {vix:.1f} -- acceptable (< 20)')
        elif vix < 25:
            failed.append(f'  [!!] VIX: {vix:.1f} -- elevated (prefer < 20 to launch)')
        else:
            failed.append(f'  [NO] VIX: {vix:.1f} -- too high (need < 20 to launch)')
    else:
        failed.append('  [??] VIX: unknown')

    # VIX trend
    if trend == 'falling':
        passed.append('  [OK] VIX trend: falling (fear declining -- ideal entry)')
    elif trend == 'flat':
        passed.append('  [OK] VIX trend: flat (stable -- acceptable)')
    elif trend == 'rising':
        failed.append('  [NO] VIX trend: rising (fear increasing -- wait)')
    else:
        failed.append('  [??] VIX trend: unknown (CBOE unavailable)')

    # VIX vs 30d avg
    if vix is not None and avg30 is not None:
        if vix < avg30 * 0.90:
            passed.append(f'  [OK] VIX vs 30d avg: {vix:.1f} vs {avg30:.1f} (well below avg)')
        elif vix < avg30:
            passed.append(f'  [OK] VIX vs 30d avg: {vix:.1f} vs {avg30:.1f} (below avg)')
        elif vix < avg30 * 1.10:
            failed.append(f'  [!!] VIX vs 30d avg: {vix:.1f} vs {avg30:.1f} (near avg)')
        else:
            failed.append(f'  [NO] VIX vs 30d avg: {vix:.1f} vs {avg30:.1f} (above avg -- stress)')
    else:
        failed.append('  [??] VIX vs 30d avg: unknown (CBOE unavailable)')

    # Regime
    if regime == 'flow':
        passed.append('  [OK] Regime: FLOW -- best for this strategy (Sharpe 8.62)')
    elif regime == 'neutral':
        passed.append('  [OK] Regime: NEUTRAL -- solid (Sharpe 4.54)')
    elif regime == 'volatility-cautious':
        failed.append('  [!!] Regime: VOLATILITY-CAUTIOUS -- reduced entries (Sharpe 7.77)')
    elif regime == 'volatility-defensive':
        failed.append('  [NO] Regime: VOLATILITY-DEFENSIVE -- no new entries (Sharpe -1.52)')
    elif regime == 'crisis':
        failed.append('  [NO] Regime: CRISIS -- do not launch')
    else:
        failed.append('  [??] Regime: unknown')

    # Earnings season
    if earn is False:
        passed.append('  [OK] Earnings season: clear')
    elif earn is True:
        failed.append('  [NO] Earnings season: active -- gap risk elevated')
    else:
        failed.append('  [??] Earnings season: unknown')

    # Fed meeting
    if fed is False:
        passed.append('  [OK] Fed meeting: none within 3 days')
    elif fed is True:
        failed.append('  [NO] Fed meeting: within 3 days -- FOMC gap risk')
    else:
        failed.append('  [??] Fed meeting: unknown')

    # Manual checks (informational only -- not scored)
    passed.append('  [--] SPY vs 50d MA: verify manually (need above for uptrend)')
    passed.append('  [--] SPY vs 20d MA: verify manually (overnight strategy filter)')

    auto_passed = [p for p in passed if '[OK]' in p]
    auto_failed = [f for f in failed if '[NO]' in f or '[!!]' in f]
    auto_total  = len(auto_passed) + len(auto_failed)
    score = int(len(auto_passed) / auto_total * 100) if auto_total > 0 else 0
    return score, passed, failed


def print_scorecard(portfolio, log_entries, as_json=False):
    metrics    = compute_paper_metrics(portfolio, log_entries)
    conditions = get_market_conditions()
    bot_score,  bot_passed,  bot_failed  = score_bot_health(metrics)
    mkt_score,  mkt_passed,  mkt_failed  = score_market_conditions(conditions)
    combined = (bot_score + mkt_score) // 2

    if bot_score == 100 and mkt_score >= 80:
        verdict = "GO  -- Bot ready + prime market window"
        detail  = "All health checks pass AND market conditions are excellent. Ideal launch moment."
    elif bot_score == 100 and mkt_score >= 60:
        verdict = "BOT READY -- but market window is marginal"
        detail  = "Bot is validated. Wait for VIX to drop or regime to improve before launching."
    elif bot_score >= 80 and mkt_score >= 80:
        verdict = "ALMOST -- fix remaining bot checks"
        detail  = "Market window is great but bot has unresolved checks. Close the gaps quickly."
    elif bot_score < 80:
        verdict = "NOT YET -- continue paper trading"
        detail  = "Bot not sufficiently validated. Keep paper trading regardless of market conditions."
    else:
        verdict = "WAIT -- poor market conditions"
        detail  = "Even if bot were ready, these market conditions are unfavorable for launch."

    if as_json:
        print(json.dumps({
            'timestamp': datetime.now().isoformat(),
            'verdict': verdict, 'bot_score': bot_score,
            'market_score': mkt_score, 'combined': combined,
            'metrics': metrics, 'conditions': conditions,
        }, indent=2))
        return

    now = datetime.now().strftime('%Y-%m-%d %H:%M ET')
    sep = '=' * 62
    print(f"\n{sep}")
    print(f"  LIVE TRADING READINESS SCORECARD  --  {now}")
    print(sep)
    print(f"  Verdict:  {verdict}")
    print(f"  {detail}")
    print(f"\n  Bot health score:      {bot_score:>3}%")
    print(f"  Market conditions:     {mkt_score:>3}%")
    print(f"  Combined:              {combined:>3}%")

    bot_status = 'PASS' if bot_score == 100 else 'FAIL'
    print(f"\n  -- Bot Health [{bot_status}] ----------------------------------------")
    for line in bot_passed + bot_failed:
        print(line)
    print(f"\n  Paper stats: {metrics['total_trades']} trades"
          f" | {metrics['wins']}W / {metrics['losses']}L"
          f" | P&L ${metrics['total_pnl']:+,.0f}"
          f" | Sharpe {metrics['sharpe']:.2f}")

    mkt_status = 'PRIME' if mkt_score >= 80 else ('OK' if mkt_score >= 60 else 'POOR')
    print(f"\n  -- Market Conditions [{mkt_status}] -- Prime Launch Window? --------")
    for line in mkt_passed + mkt_failed:
        print(line)

    if mkt_score >= 80:
        print("\n  >>> PRIME WINDOW: Excellent conditions for launch")
    elif mkt_score >= 60:
        print("\n  >>> ACCEPTABLE WINDOW: Address warnings before launching")
    elif mkt_score >= 40:
        print("\n  >>> MARGINAL WINDOW: Significant headwinds -- consider waiting")
    else:
        print("\n  >>> POOR WINDOW: Do not launch in these conditions")

    print(f"\n  -- Manual checks before going live --------------------------")
    for item in [
        "Schwab account has sufficient buying power",
        "Options trading approved (Level 2+)",
        "No earnings in portfolio within 48hrs",
        "SPY above 20d AND 50d EMA",
        "End-to-end dry run completed (#7)",
        "Anomaly notifications configured (#4)",
        "Server health monitoring active (#8)",
    ]:
        print(f"  [ ] {item}")
    print(f"{sep}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--portfolio', default='paper_portfolio.json')
    parser.add_argument('--log',       default='bot_log.jsonl')
    args = parser.parse_args()
    portfolio   = load_paper_portfolio(args.portfolio)
    log_entries = load_bot_log(args.log)
    print_scorecard(portfolio, log_entries, as_json=args.json)