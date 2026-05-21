"""
sweep_overnight.py — Overnight Drift Strategy Parameter Sweep

Tests regime-aware ETF allocations and position sizing for the overnight
drift strategy. Buys index ETFs at close, sells at next open.

Data-driven approach — sweep finds optimal:
  - Which ETFs to hold per regime (SPY, QQQ, IWM, GLD, TLT)
  - Position size as % of free cash per regime
  - Whether each regime should participate at all

Usage:
    python sweep_overnight.py              # 5yr sweep, all regimes
    python sweep_overnight.py --years 3    # faster 3yr run

Output:
    - Per-regime P&L, Sharpe, win rate, avg overnight return
    - Best ETF and size per regime
    - Overall recommendation table
"""

import sys
import io
import time
import itertools
import warnings
import argparse
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

try:
    from auth import authenticate
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

STARTING_CAPITAL = 25_000.0
SPREAD_COST_PCT  = 0.0005   # 0.05% round-trip spread

# ETFs to test as overnight vehicles
OVERNIGHT_ETFS = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'EWJ', 'EWY', 'EFA', 'EEM', 'FXI']

# Position sizes to test (% of free cash)
POSITION_SIZES = [0.10, 0.20, 0.30, 0.40, 0.50]

# Regimes to sweep
REGIMES = ['flow', 'neutral', 'volatility-cautious']
# vol-defensive and crisis excluded — no overnight position in stressed markets

# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_price_history(tickers, years=5):
    try:
        import yfinance as yf
    except ImportError:
        print("❌ yfinance not installed")
        sys.exit(1)

    history = {}
    try:
        raw = yf.download(tickers, period=f"{years}y", auto_adjust=True,
                          progress=False, threads=True)
        if raw.empty:
            return {}
        if isinstance(raw.columns, pd.MultiIndex):
            for sym in tickers:
                try:
                    df = raw.xs(sym, axis=1, level=1).copy()
                    df.columns = [c.lower() for c in df.columns]
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    df = df.dropna(subset=['close'])
                    if len(df) > 30:
                        history[sym] = df
                except Exception:
                    pass
        else:
            raw.columns = [c.lower() for c in raw.columns]
            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            raw = raw.dropna(subset=['close'])
            if len(raw) > 30:
                history[tickers[0]] = raw
    except Exception as e:
        print(f"  Batch error: {e}")

    print(f"  📊 Fetched {len(history)}/{len(tickers)} tickers")
    return history


def fetch_vix_history():
    try:
        import requests
        url  = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv'
        resp = requests.get(url, timeout=15)
        df   = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.strip().lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        print(f"  ✅ VIX history: {len(df)} days")
        return df
    except Exception as e:
        print(f"  ⚠️  VIX fetch failed: {e}")
        return pd.DataFrame()


def get_vix_on_date(vix_df, date):
    if vix_df.empty:
        return 20.0
    try:
        idx = vix_df.index.searchsorted(pd.Timestamp(date))
        idx = min(idx, len(vix_df) - 1)
        return float(vix_df.iloc[idx]['close'])
    except Exception:
        return 20.0


def get_vix_30d_avg(vix_df, date):
    if vix_df.empty:
        return 20.0
    try:
        ts     = pd.Timestamp(date)
        subset = vix_df[(vix_df.index >= ts - timedelta(days=30)) & (vix_df.index <= ts)]
        return float(subset['close'].mean()) if len(subset) > 0 else 20.0
    except Exception:
        return 20.0


def detect_regime(vix, vix_30d, spy_price=0.0, spy_ema50=0.0):
    if vix_30d <= 0:
        return 'neutral'
    if vix > 40:
        return 'crisis'
    ratio = vix / vix_30d
    if ratio > 1.15:
        spy_below = (spy_price > 0 and spy_ema50 > 0 and spy_price < spy_ema50)
        if ratio > 1.30 or spy_below:
            return 'volatility-defensive'
        return 'volatility-cautious'
    elif ratio < 0.85:
        return 'flow'
    return 'neutral'


# ── Pre-compute overnight returns ─────────────────────────────────────────────

def compute_overnight_returns(history):
    """
    For each ETF, compute overnight return = (next_open - close) / close.
    This is the return from buying at close and selling at next open.
    Includes a spread cost on both legs.
    Also computes per-day-of-week breakdown.
    """
    overnight = {}
    for sym, df in history.items():
        if 'open' not in df.columns:
            continue
        closes  = df['close']
        opens   = df['open'].shift(-1)   # next day's open
        raw_ret = (opens - closes) / closes
        net_ret = raw_ret - (SPREAD_COST_PCT * 2)
        overnight[sym] = net_ret.dropna()
    return overnight


def compute_dow_returns(history):
    """
    Returns per-ETF per-day-of-week overnight return stats.
    Day 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday
    Friday overnight = weekend hold (Fri close → Mon open) — typically negative.
    """
    DOW_NAMES = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri'}
    results = {}
    for sym, df in history.items():
        if 'open' not in df.columns:
            continue
        closes  = df['close']
        opens   = df['open'].shift(-1)
        raw_ret = (opens - closes) / closes
        net_ret = (raw_ret - SPREAD_COST_PCT * 2).dropna()
        dow_stats = {}
        for day_num, day_name in DOW_NAMES.items():
            day_rets = net_ret[net_ret.index.dayofweek == day_num]
            if len(day_rets) < 10:
                continue
            dow_stats[day_name] = {
                'avg':      round(day_rets.mean() * 100, 4),
                'win_pct':  round((day_rets > 0).mean() * 100, 1),
                'n':        len(day_rets),
                'sharpe':   round(day_rets.mean() / day_rets.std() * (252**0.5), 3)
                            if day_rets.std() > 0 else 0,
            }
        results[sym] = dow_stats
    return results


# ── Single regime sweep ───────────────────────────────────────────────────────

def sweep_regime_etf(regime, etf, size_pct, history, vix_df, overnight_returns, day_filter=None):
    """
    Simulate overnight strategy for one regime/ETF/size combination.
    Only enters on days when regime matches.
    Returns metrics dict.
    """
    spy_hist = history.get('SPY', pd.DataFrame())
    dates    = sorted(spy_hist.index) if not spy_hist.empty else []
    if not dates:
        return None

    ov_returns = overnight_returns.get(etf)
    if ov_returns is None or ov_returns.empty:
        return None

    capital  = STARTING_CAPITAL
    cash     = capital
    trades   = []

    for date in dates:
        vix     = get_vix_on_date(vix_df, date)
        vix_30d = get_vix_30d_avg(vix_df, date)

        spy_slice = spy_hist[spy_hist.index <= pd.Timestamp(date)].tail(50)
        spy_price = float(spy_slice['close'].iloc[-1]) if not spy_slice.empty else 0.0
        spy_ema50 = float(spy_slice['close'].ewm(span=50, adjust=False).mean().iloc[-1]) \
                    if len(spy_slice) >= 10 else 0.0

        day_regime = detect_regime(vix, vix_30d, spy_price, spy_ema50)

        # Only trade on matching regime days
        if day_regime != regime:
            continue

        # Day-of-week filter
        if day_filter is not None and date.dayofweek not in day_filter:
            continue

        # Skip if ETF has no overnight return for this date
        if date not in ov_returns.index:
            continue

        ov_ret = float(ov_returns.loc[date])
        if pd.isna(ov_ret):
            continue

        # Position: size_pct of current cash
        position_value = cash * size_pct
        pnl            = position_value * ov_ret
        cash          += pnl

        trades.append({
            'date':   date,
            'pnl':    pnl,
            'ret':    ov_ret,
            'regime': day_regime,
        })

    if len(trades) < 10:
        return None

    pnls     = [t['pnl'] for t in trades]
    rets     = [t['ret'] for t in trades]
    wins     = [p for p in pnls if p > 0]
    win_rate = len(wins) / len(trades) * 100

    s = pd.Series(rets)
    sharpe = (s.mean() / s.std() * (252 ** 0.5)) if s.std() > 0 else 0

    total_pnl  = sum(pnls)
    avg_ret    = s.mean() * 100
    avg_pnl    = total_pnl / len(trades)

    # Max drawdown
    peak, maxdd, running = 0, 0.0, 0.0
    for p in pnls:
        running += p
        peak = max(peak, running)
        dd   = (peak - running) / max(peak, 1) * 100
        maxdd = max(maxdd, dd)

    return {
        'sharpe':    round(sharpe, 3),
        'win_rate':  round(win_rate, 1),
        'trades':    len(trades),
        'total_pnl': round(total_pnl, 2),
        'avg_ret':   round(avg_ret, 4),
        'avg_pnl':   round(avg_pnl, 2),
        'maxdd':     round(maxdd, 2),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Overnight Drift Strategy Sweep')
    parser.add_argument('--years', default=5, type=int)
    args = parser.parse_args()

    print("🔌 Authenticating with Schwab...")
    client, paper = authenticate()
    print("✅ Connected\n")

    tickers = list(set(OVERNIGHT_ETFS + ['SPY']))
    print(f"📥 Fetching {args.years}yr price history ({len(tickers)} tickers)...")
    history = fetch_price_history(tickers, years=args.years)

    print("\n📈 Fetching VIX history...")
    vix_df = fetch_vix_history()

    print("\n⚡ Pre-computing overnight returns (close → next open)...")
    overnight_returns = compute_overnight_returns(history)
    for sym, rets in overnight_returns.items():
        avg = rets.mean() * 100
        pos = (rets > 0).mean() * 100
        print(f"  {sym:<6} avg overnight: {avg:+.3f}%  positive: {pos:.1f}%  n={len(rets)}")

    # ── Day-of-week breakdown ────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  DAY-OF-WEEK OVERNIGHT RETURNS (all regimes, after spread)")
    print(f"{'='*72}")
    print(f"  {'ETF':<6}  {'Mon':>8}  {'Tue':>8}  {'Wed':>8}  {'Thu':>8}  {'Fri':>8}  Best days")
    print(f"  {'-'*68}")

    dow_results = compute_dow_returns(history)
    positive_days = {}   # ETF -> list of day numbers with positive avg
    for sym in OVERNIGHT_ETFS:
        stats = dow_results.get(sym, {})
        if not stats:
            continue
        avgs = {d: stats[d]['avg'] for d in stats}
        row  = f"  {sym:<6}"
        best = []
        for day_name in ['Mon','Tue','Wed','Thu','Fri']:
            if day_name in avgs:
                val  = avgs[day_name]
                mark = '✅' if val > 0 else '  '
                row += f"  {mark}{val:>+5.3f}%"
                if val > 0:
                    best.append(day_name)
            else:
                row += f"  {'N/A':>8}"
        row += f"  {', '.join(best) if best else 'none'}"
        print(row)
        # Map day names to numbers for filter
        day_map = {'Mon':0,'Tue':1,'Wed':2,'Thu':3,'Fri':4}
        positive_days[sym] = [day_map[d] for d in best]

    # ── Sweep — all nights ────────────────────────────────────────────────────
    all_results = defaultdict(list)

    for regime in REGIMES:
        print(f"\n{'='*72}")
        print(f"  REGIME: {regime.upper()}")
        print(f"{'='*72}")
        print(f"  {'ETF':<6}  {'Size':>5}  {'Sharpe':>7}  {'WinR':>5}  "
              f"{'Trades':>6}  {'AvgRet':>7}  {'AvgPnL':>8}  {'MaxDD':>6}  {'TotalPnL':>10}")
        print(f"  {'-'*70}")

        regime_results = []
        for etf, size in itertools.product(OVERNIGHT_ETFS, POSITION_SIZES):
            r = sweep_regime_etf(regime, etf, size, history, vix_df, overnight_returns)
            if not r:
                continue
            r.update(regime=regime, etf=etf, size=size)
            regime_results.append(r)
            all_results[regime].append(r)

            best = max((x['sharpe'] for x in regime_results), default=0)
            mark = ' ◀' if r['sharpe'] == best else ''
            print(f"  {etf:<6}  {size:>4.0%}  {r['sharpe']:>7.3f}  "
                  f"{r['win_rate']:>4.1f}%  {r['trades']:>6}  "
                  f"{r['avg_ret']:>6.3f}%  ${r['avg_pnl']:>7.2f}  "
                  f"{r['maxdd']:>5.1f}%  ${r['total_pnl']:>9,.0f}{mark}")

    # ── Targeted sweep — positive days only ─────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  TARGETED SWEEP — POSITIVE DAYS ONLY (per ETF)")
    print(f"{'='*72}")

    targeted_results = defaultdict(list)
    for regime in REGIMES:
        print(f"\n  Regime: {regime.upper()}")
        print(f"  {'ETF':<6}  {'Days':<15}  {'Size':>5}  {'Sharpe':>7}  {'WinR':>5}  {'AvgRet':>7}  {'TotalPnL':>10}")
        print(f"  {'-'*65}")
        for etf in OVERNIGHT_ETFS:
            day_filter = positive_days.get(etf)
            if not day_filter:
                print(f"  {etf:<6}  {'no positive days':<15}  {'—':>5}")
                continue
            day_names = [['Mon','Tue','Wed','Thu','Fri'][d] for d in day_filter]
            best_r = None
            for size in POSITION_SIZES:
                r = sweep_regime_etf(regime, etf, size, history, vix_df,
                                     overnight_returns, day_filter=day_filter)
                if not r:
                    continue
                r.update(regime=regime, etf=etf, size=size)
                targeted_results[regime].append(r)
                if best_r is None or r['sharpe'] > best_r['sharpe']:
                    best_r = r
            if best_r:
                mark = ' ✅' if best_r['sharpe'] > 0 else ''
                print(f"  {etf:<6}  {'+'.join(day_names):<15}  {best_r['size']:>4.0%}  "
                      f"{best_r['sharpe']:>7.3f}  {best_r['win_rate']:>4.1f}%  "
                      f"{best_r['avg_ret']:>6.3f}%  ${best_r['total_pnl']:>9,.0f}{mark}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  OVERNIGHT STRATEGY RECOMMENDATIONS")
    print(f"{'='*72}")
    print(f"  {'Regime':<22}  {'ETF':<6}  {'Size':>5}  {'Sharpe':>7}  "
          f"{'WinR':>5}  {'AvgRet':>7}  {'TotalPnL':>10}")
    print(f"  {'-'*70}")

    recommendations = {}
    for regime in REGIMES:
        results = all_results[regime]
        if not results:
            print(f"  {regime:<22}  {'N/A':<6}  {'N/A':>5}  {'N/A':>7}  "
                  f"{'N/A':>5}  {'N/A':>7}  {'N/A':>10}")
            continue
        best = max(results, key=lambda x: x['sharpe'])
        recommendations[regime] = best
        print(f"  {regime:<22}  {best['etf']:<6}  {best['size']:>4.0%}  "
              f"{best['sharpe']:>7.3f}  {best['win_rate']:>4.1f}%  "
              f"{best['avg_ret']:>6.3f}%  ${best['total_pnl']:>9,.0f}")

    # vol-defensive and crisis
    for r in ['volatility-defensive', 'crisis']:
        print(f"  {r:<22}  {'SKIP':<6}  {'N/A':>5}  {'N/A':>7}  "
              f"{'N/A':>5}  {'N/A':>7}  {'N/A':>10}  (capital preservation)")

    print(f"\n{'='*72}")
    print(f"  RAW OVERNIGHT DRIFT BY ETF (all regimes combined)")
    print(f"{'='*72}")
    print(f"  {'ETF':<6}  {'AvgRet':>7}  {'WinRate':>8}  {'Ann Drift':>10}  Note")
    print(f"  {'-'*55}")
    for sym in OVERNIGHT_ETFS:
        rets = overnight_returns.get(sym)
        if rets is None:
            continue
        avg    = rets.mean() * 100
        pos    = (rets > 0).mean() * 100
        annual = ((1 + rets.mean()) ** 252 - 1) * 100
        note   = '← strongest' if sym == max(
            OVERNIGHT_ETFS,
            key=lambda s: overnight_returns.get(s, pd.Series()).mean()
            if s in overnight_returns else -999
        ) else ''
        print(f"  {sym:<6}  {avg:>6.3f}%  {pos:>7.1f}%  {annual:>9.1f}%  {note}")

    print(f"\n✅ Sweep complete — copy recommendations into overnight strategy config")


if __name__ == '__main__':
    main()