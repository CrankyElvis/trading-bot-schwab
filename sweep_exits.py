"""
sweep_exits.py  —  Exit Parameter Sweep (standalone, data fetched ONCE)

Fetches price history and VIX once, then runs all 80 parameter combinations
in memory. Estimated runtime: 15-25 minutes (vs 4-7 hours for backtester --sweep)

Usage:
    python sweep_exits.py                    # 5yr, $25k, all 80 combos
    python sweep_exits.py --years 3          # faster 3yr run
    python sweep_exits.py --years 5 --top 15 # show top 15 results

Output:
    - Live matrix as combos complete
    - Top 10 by Sharpe
    - Top 10 by lowest opportunity cost
    - Comparison vs current settings (5%/15%/5d)
"""

import os
import sys
import io
import time
import json
import math
import argparse
import warnings
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

STARTING_CAPITAL  = 25_000.0
SPREAD_COST_PCT   = 0.0005
MAX_POSITIONS     = 5
BASE_POSITION_PCT = 0.20
MAX_CANDIDATES    = 4
MIN_SCORE         = 0.50

# Sweep grid
STOP_LOSSES  = [0.03, 0.05, 0.07, 0.10]
TAKE_PROFITS = [0.10, 0.15, 0.20, 0.25, 0.30]
MAX_HOLDS    = [3, 5, 7, 10]

# ── Imports from bot ──────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))

try:
    from auth import authenticate
    from flow_momentum import WEIGHTS as CURRENT_WEIGHTS, get_max_candidates, compute_adx
    from exit_manager import (
        CRISIS_STOP_PCT, CRISIS_PROFIT_PCT, CRISIS_HOLD_DAYS, dynamic_take_profit
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# ── Inline universe + data fetch (same as backtester) ────────────────────────

try:
    from data_collector import DEFAULT_UNIVERSE
    UNIVERSE = list(DEFAULT_UNIVERSE)
except Exception:
    UNIVERSE = []

try:
    from flow_momentum import SCAN_UNIVERSE
    EXTRA = [s for s in SCAN_UNIVERSE if s not in UNIVERSE]
    ALL_TICKERS = UNIVERSE + EXTRA
except Exception:
    ALL_TICKERS = UNIVERSE

# Add macro tickers
for t in ['SPY','QQQ','IWM','VIXY','TLT','HYG','LQD','USO','GLD']:
    if t not in ALL_TICKERS:
        ALL_TICKERS.append(t)

# ── Data fetching (once) ──────────────────────────────────────────────────────

def fetch_vix_history():
    try:
        import requests
        url = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv'
        resp = requests.get(url, timeout=15)
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.strip().lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        print(f"  ✅ VIX history: {len(df)} days")
        return df
    except Exception as e:
        print(f"  ⚠️  VIX fetch failed: {e} — using VIXY fallback")
        return pd.DataFrame()

def fetch_price_history(tickers, years=5):
    try:
        import yfinance as yf
    except ImportError:
        print("❌ yfinance not installed")
        sys.exit(1)

    period = f"{years}y"
    history = {}
    batch_size = 50
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]

    for batch in batches:
        try:
            raw = yf.download(batch, period=period, auto_adjust=True,
                              progress=False, threads=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                for sym in batch:
                    try:
                        df = raw.xs(sym, axis=1, level=1).copy()
                        df.columns = [c.lower() for c in df.columns]
                        df = df.dropna(subset=['close'])
                        if len(df) > 30:
                            history[sym] = df
                    except Exception:
                        pass
            else:
                raw.columns = [c.lower() for c in raw.columns]
                raw = raw.dropna(subset=['close'])
                if len(raw) > 30:
                    history[batch[0]] = raw
        except Exception as e:
            print(f"  Batch error: {e}")

    # Single-ticker fallback
    missing = [t for t in tickers if t not in history]
    for sym in missing:
        try:
            df = yf.Ticker(sym).history(period=period)
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.dropna(subset=['close'])
            if len(df) > 30:
                history[sym] = df
        except Exception:
            pass

    print(f"  📊 Fetched {len(history)}/{len(tickers)} tickers")
    return history

def get_vix_on_date(vix_df, date):
    if vix_df.empty:
        return 20.0
    try:
        ts = pd.Timestamp(date)
        idx = vix_df.index.searchsorted(ts)
        if idx >= len(vix_df):
            idx = len(vix_df) - 1
        return float(vix_df.iloc[idx]['close'])
    except Exception:
        return 20.0

def get_vix_30d_avg(vix_df, date):
    if vix_df.empty:
        return 20.0
    try:
        ts = pd.Timestamp(date)
        cutoff = ts - timedelta(days=30)
        subset = vix_df[(vix_df.index >= cutoff) & (vix_df.index <= ts)]
        return float(subset['close'].mean()) if len(subset) > 0 else 20.0
    except Exception:
        return 20.0

# ── Regime detection ──────────────────────────────────────────────────────────

def detect_regime(vix, vix_30d):
    vixy_ratio = vix / vix_30d if vix_30d > 0 else 1.0
    if vix >= 35:
        return 'crisis'
    if vixy_ratio >= 1.15:
        return 'volatility'
    if vixy_ratio <= 0.90 and vix < 18:
        return 'flow'
    return 'neutral'

# ── Slippage ──────────────────────────────────────────────────────────────────

_ETF_SYMS = {'SPY','QQQ','IWM','GLD','TLT','HYG','LQD','USO','GDX','VTI',
             'XLF','XLE','XLK','XLV','VIXY','VXX','UVXY','SQQQ','SPXU'}
_LARGE_CAP = {'AAPL','MSFT','AMZN','GOOGL','META','NVDA','TSLA','JPM',
               'V','MA','UNH','JNJ','XOM','CVX','ABBV','LLY','AVGO'}

def apply_slippage(price, symbol, is_buy):
    if symbol in _ETF_SYMS:   slip = 0.0001
    elif symbol in _LARGE_CAP: slip = 0.0005
    else:                       slip = 0.0010
    return price * (1 + slip) if is_buy else price * (1 - slip)

# ── Simplified single-pass backtest (no print output) ────────────────────────

@dataclass
class Pos:
    symbol:      str
    shares:      float
    entry_price: float
    entry_date:  object
    cost_basis:  float
    signals:     dict = field(default_factory=dict)
    regime:      str  = 'neutral'

@dataclass
class Trade:
    symbol:      str
    action:      str
    pnl:         float
    hold_days:   int
    exit_reason: str
    regime:      str

def score_symbol(symbol, date, history, spy_hist, vix):
    """Simplified scoring — same signal proxies as main backtester."""
    df = history.get(symbol)
    if df is None or date not in df.index:
        return 0.0, {}

    loc = df.index.get_loc(date)
    if loc < 30:
        return 0.0, {}

    closes  = df['close'].iloc[max(0, loc-60):loc+1]
    volumes = df['volume'].iloc[max(0, loc-60):loc+1] if 'volume' in df else pd.Series()
    price   = float(closes.iloc[-1])

    sigs = {}

    # market_tide
    spy = history.get('SPY')
    if spy is not None and date in spy.index:
        sp_loc = spy.index.get_loc(date)
        if sp_loc >= 20:
            sp_c = spy['close'].iloc[sp_loc-20:sp_loc+1]
            ema  = sp_c.ewm(span=20).mean().iloc[-1]
            sigs['market_tide'] = 1.0 if spy['close'].iloc[sp_loc] > ema else 0.0

    # sector_tide (price momentum 20d)
    if len(closes) >= 20:
        ret20 = (closes.iloc[-1] / closes.iloc[-20] - 1)
        sigs['sector_tide'] = min(1.0, max(0.0, (ret20 + 0.05) / 0.10))

    # price_rvol
    if len(closes) >= 20 and len(volumes) >= 20:
        ret5  = (closes.iloc[-1] / closes.iloc[-5] - 1)
        avgv  = volumes.iloc[-20:-1].mean()
        todv  = volumes.iloc[-1]
        rvol  = todv / avgv if avgv > 0 else 1.0
        sigs['price_rvol'] = min(1.0, max(0.0, (ret5 * rvol + 0.02) / 0.05))

    # Use 0.5 for signals requiring live data
    for s in ['sweep_flow', 'dark_pool', 'politician', 'insider', 'gex', 'etf_flow']:
        sigs[s] = 0.5

    weights = CURRENT_WEIGHTS
    score = sum(weights.get(k, 0) * v for k, v in sigs.items())
    return round(score, 4), sigs


def run_single(history, vix_df, capital, stop_pct, tp_pct, hold_days):
    """Run one backtest iteration with given exit params. Returns metrics dict."""
    dates = sorted(history.get('SPY', pd.DataFrame()).index)
    if not dates:
        return None

    positions = {}
    cash      = capital
    total_fees = 0.0
    trades    = []
    last_regime = 'neutral'

    spy_hist = history.get('SPY', pd.DataFrame())
    UNIVERSE_LOCAL = [t for t in ALL_TICKERS if t in history]

    for date in dates:
        prices = {}
        for sym in UNIVERSE_LOCAL:
            df = history[sym]
            if date in df.index:
                prices[sym] = float(df.loc[date, 'close'])

        if not prices.get('SPY'):
            continue

        vix     = get_vix_on_date(vix_df, date)
        vix_30d = get_vix_30d_avg(vix_df, date)
        regime  = detect_regime(vix, vix_30d)

        # ADX gate
        spy_slice = spy_hist[spy_hist.index <= pd.Timestamp(date)].tail(60)
        spy_adx   = compute_adx(spy_slice)

        # Exit open positions
        to_exit = []
        for sym, pos in positions.items():
            price = prices.get(sym, 0)
            if price <= 0:
                continue
            pnl_pct  = (price - pos.entry_price) / pos.entry_price
            held     = (pd.Timestamp(date) - pd.Timestamp(pos.entry_date)).days

            # Regime flip half-exit
            if regime in ('volatility', 'crisis') and last_regime in ('neutral', 'flow'):
                to_exit.append((sym, price, 'regime_flip'))
                continue

            # Use sweep take-profit param directly
            profit_target = tp_pct

            if pnl_pct <= -stop_pct:
                to_exit.append((sym, price, 'stop_loss'))
            elif pnl_pct >= profit_target:
                to_exit.append((sym, price, 'take_profit'))
            elif held >= hold_days:
                to_exit.append((sym, price, 'time_stop'))

        for sym, exit_price, reason in to_exit:
            if sym not in positions:
                continue
            pos  = positions.pop(sym)
            ep   = apply_slippage(exit_price, sym, False)
            val  = pos.shares * ep
            fee  = val * SPREAD_COST_PCT
            pnl  = (val - fee) - pos.cost_basis
            held = (pd.Timestamp(date) - pd.Timestamp(pos.entry_date)).days
            cash += val - fee
            total_fees += fee
            trades.append(Trade(sym, 'SELL', pnl, held, reason, regime))

        last_regime = regime

        # New entries
        if regime == 'crisis' or len(positions) >= MAX_POSITIONS:
            continue

        # ADX gate
        max_new = get_max_candidates(cash + sum(
            positions[s].shares * prices.get(s, positions[s].entry_price)
            for s in positions
        ))
        if spy_adx > 0 and spy_adx < 20:
            max_new = max(1, max_new // 2)

        if max_new <= 0:
            continue

        scores = {}
        for sym in UNIVERSE_LOCAL:
            if sym in positions or sym not in prices:
                continue
            sc, sigs = score_symbol(sym, date, history, spy_hist, vix)
            if sc >= MIN_SCORE:
                scores[sym] = (sc, sigs)

        top = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)[:max_new]
        idle = cash * 0.90
        pos_dollars = idle * BASE_POSITION_PCT

        for sym, (sc, sigs) in top:
            price = prices.get(sym, 0)
            if price <= 0 or pos_dollars > cash:
                continue
            ep    = apply_slippage(price, sym, True)
            fee   = pos_dollars * SPREAD_COST_PCT
            shares = (pos_dollars - fee) / ep
            cash  -= pos_dollars
            total_fees += fee
            positions[sym] = Pos(sym, shares, ep, date, pos_dollars - fee,
                                 sigs, regime)

    # Liquidate remaining
    final_date = dates[-1]
    final_prices = {sym: float(history[sym].loc[final_date, 'close'])
                    for sym in positions if final_date in history.get(sym, pd.DataFrame()).index}

    for sym, pos in list(positions.items()):
        price = final_prices.get(sym, pos.entry_price)
        ep    = apply_slippage(price, sym, False)
        val   = pos.shares * ep
        fee   = val * SPREAD_COST_PCT
        pnl   = (val - fee) - pos.cost_basis
        held  = (pd.Timestamp(final_date) - pd.Timestamp(pos.entry_date)).days
        cash += val - fee
        total_fees += fee
        trades.append(Trade(sym, 'SELL', pnl, held, 'liquidation', 'neutral'))

    # Metrics
    final_value  = cash
    total_return = (final_value - capital) / capital
    n_days       = len(dates)
    years        = n_days / 252
    annual       = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    sells = [t for t in trades if t.action == 'SELL']
    n     = len(sells)
    if n == 0:
        return None

    # Daily returns for Sharpe
    daily_vals = []
    running    = capital
    for t in sells:
        running += t.pnl
        daily_vals.append(running)

    if len(daily_vals) > 1:
        rets   = pd.Series(daily_vals).pct_change().dropna()
        sharpe = (rets.mean() / rets.std() * (252 ** 0.5)) if rets.std() > 0 else 0
    else:
        sharpe = 0

    wins     = [t for t in sells if t.pnl > 0]
    win_rate = len(wins) / n * 100

    # Max drawdown
    peak = capital
    maxdd = 0.0
    running = capital
    for t in sells:
        running += t.pnl
        peak = max(peak, running)
        dd   = (peak - running) / peak * 100
        maxdd = max(maxdd, dd)

    avg_hold = sum(t.hold_days for t in sells) / n

    # Opportunity cost: flow-regime trades that exited at stop or time_stop
    opp_exits = [t for t in sells
                 if t.exit_reason in ('stop_loss', 'time_stop')
                 and t.regime == 'flow']
    opp_pct   = len(opp_exits) / n * 100

    return {
        'sharpe':       round(sharpe, 3),
        'annual':       round(annual * 100, 2),
        'maxdd':        round(maxdd, 2),
        'win_rate':     round(win_rate, 1),
        'trades':       n,
        'avg_hold':     round(avg_hold, 1),
        'opp_pct':      round(opp_pct, 1),
        'final_value':  round(final_value, 2),
    }


# ── Main sweep ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Exit Parameter Sweep')
    parser.add_argument('--years',   default=5,     type=int)
    parser.add_argument('--capital', default=25000, type=float)
    parser.add_argument('--top',     default=10,    type=int)
    args = parser.parse_args()

    print("🔌 Authenticating with Schwab...")
    client = authenticate()
    print("✅ Connected\n")

    print(f"📥 Fetching price history ({args.years}yr, {len(ALL_TICKERS)} tickers)...")
    print("   (This happens ONCE — all 80 combos share this data)\n")
    history = fetch_price_history(ALL_TICKERS, years=args.years)

    print("\n📈 Fetching VIX history...")
    vix_df = fetch_vix_history()

    combos = list(itertools.product(STOP_LOSSES, TAKE_PROFITS, MAX_HOLDS))
    print(f"\n{'='*85}")
    print(f"  EXIT PARAMETER SWEEP  —  {len(combos)} combinations")
    print(f"  Stop: {[f'{s:.0%}' for s in STOP_LOSSES]}")
    print(f"  TP:   {[f'{t:.0%}' for t in TAKE_PROFITS]}")
    print(f"  Hold: {MAX_HOLDS} days")
    print(f"{'='*85}")
    print(f"  {'Stop':>5}  {'TP':>5}  {'Hold':>4}  {'Sharpe':>7}  {'Ann%':>7}  "
          f"{'MaxDD':>6}  {'WinR':>5}  {'Trades':>6}  {'AvgHold':>7}  {'OppCost':>8}")
    print(f"  {'-'*83}")

    results = []
    t0 = time.time()

    for i, (stop, tp, hold) in enumerate(combos, 1):
        try:
            r = run_single(history, vix_df, args.capital, stop, tp, hold)
            if not r:
                continue
            r.update(stop=stop, tp=tp, hold=hold)
            results.append(r)

            best = max(results, key=lambda x: x['sharpe'])['sharpe']
            mark = ' ◀' if r['sharpe'] == best else ''
            eta  = (time.time() - t0) / i * (len(combos) - i)

            print(f"  {stop:>4.0%}  {tp:>4.0%}  {hold:>4}d  "
                  f"{r['sharpe']:>7.3f}  {r['annual']:>6.1f}%  "
                  f"{r['maxdd']:>5.1f}%  {r['win_rate']:>4.1f}%  "
                  f"{r['trades']:>6}  {r['avg_hold']:>6.1f}d  "
                  f"{r['opp_pct']:>7.1f}%{mark}  "
                  f"[ETA {eta/60:.0f}m]")
        except Exception as e:
            print(f"  {stop:.0%}/{tp:.0%}/{hold}d  ERROR: {e}")

    if not results:
        print("No results.")
        return

    n = args.top

    print(f"\n{'='*75}")
    print(f"  TOP {n} BY SHARPE RATIO")
    print(f"{'='*75}")
    print(f"  {'Stop':>5}  {'TP':>5}  {'Hold':>4}  {'Sharpe':>7}  {'Ann%':>7}  "
          f"{'MaxDD':>6}  {'WinR':>5}  {'OppCost':>8}")
    print(f"  {'-'*65}")
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True)[:n]:
        print(f"  {r['stop']:>4.0%}  {r['tp']:>4.0%}  {r['hold']:>4}d  "
              f"{r['sharpe']:>7.3f}  {r['annual']:>6.1f}%  "
              f"{r['maxdd']:>5.1f}%  {r['win_rate']:>4.1f}%  "
              f"{r['opp_pct']:>7.1f}%")

    print(f"\n{'='*75}")
    print(f"  TOP {n} BY LOWEST OPPORTUNITY COST (flow regime stop/time exits)")
    print(f"{'='*75}")
    print(f"  {'Stop':>5}  {'TP':>5}  {'Hold':>4}  {'Sharpe':>7}  {'Ann%':>7}  "
          f"{'OppCost':>8}  {'AvgHold':>7}")
    print(f"  {'-'*65}")
    for r in sorted(results, key=lambda x: x['opp_pct'])[:n]:
        print(f"  {r['stop']:>4.0%}  {r['tp']:>4.0%}  {r['hold']:>4}d  "
              f"{r['sharpe']:>7.3f}  {r['annual']:>6.1f}%  "
              f"{r['opp_pct']:>7.1f}%  {r['avg_hold']:>6.1f}d")

    curr = next((r for r in results
                 if abs(r['stop']-0.05)<0.001
                 and abs(r['tp']-0.15)<0.001
                 and r['hold']==5), None)
    best = max(results, key=lambda x: x['sharpe'])

    print(f"\n{'='*75}")
    print(f"  SUMMARY")
    print(f"{'='*75}")
    if curr:
        print(f"  Current (5%/15%/5d):  Sharpe={curr['sharpe']:.3f}  "
              f"Ann={curr['annual']:.1f}%  OppCost={curr['opp_pct']:.1f}%")
    print(f"  Best   ({best['stop']:.0%}/{best['tp']:.0%}/{best['hold']}d):  "
          f"Sharpe={best['sharpe']:.3f}  Ann={best['annual']:.1f}%  "
          f"OppCost={best['opp_pct']:.1f}%")
    if curr:
        print(f"  Delta:  +{best['sharpe']-curr['sharpe']:.3f} Sharpe  "
              f"  {curr['opp_pct']-best['opp_pct']:+.1f}% opportunity cost")
    elapsed = (time.time() - t0) / 60
    print(f"\n✅ Sweep complete — {len(results)}/{len(combos)} combinations in {elapsed:.1f} min")


if __name__ == '__main__':
    main()