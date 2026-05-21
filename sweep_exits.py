"""
sweep_exits.py  —  Regime-Aware Exit Parameter Sweep

Fetches price history and VIX ONCE, then sweeps all parameter combinations
per regime. Each regime gets its own optimal stop/TP/hold settings.

Regimes swept independently:
  flow               — momentum regime (SPY ratio < 0.90, VIX < 18)
  neutral            — default regime
  volatility-cautious — ratio 1.15-1.30, SPY above 50d EMA
  volatility-defensive — ratio > 1.30 OR SPY below 50d EMA

Crisis excluded — too few days to sweep meaningfully (capital preservation mode).

Usage:
    python sweep_exits.py                    # 5yr, all combos, all regimes
    python sweep_exits.py --years 3          # faster 3yr run
    python sweep_exits.py --top 10           # show top 10 per regime
    python sweep_exits.py --regime neutral   # sweep one regime only
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

# Per-regime sweep grids — tuned to each regime's characteristics
REGIME_GRIDS = {
    'flow': {
        'stop':  [0.03, 0.05, 0.07, 0.10],
        'tp':    [0.15, 0.20, 0.25, 0.30, 0.35],
        'hold':  [3, 5, 7, 10, 14],
        'note':  'Momentum regime — let winners run, wide TP',
    },
    'neutral': {
        'stop':  [0.03, 0.05, 0.07, 0.10],
        'tp':    [0.10, 0.15, 0.20, 0.25],
        'hold':  [3, 5, 7, 10],
        'note':  'Default regime — standard ranges',
    },
    'volatility-cautious': {
        'stop':  [0.02, 0.03, 0.04, 0.05],
        'tp':    [0.08, 0.10, 0.12, 0.15],
        'hold':  [2, 3, 5, 7],
        'note':  'Elevated vol, SPY above EMA — tighter, faster exits',
    },
    'volatility-defensive': {
        'stop':  [0.02, 0.03, 0.04, 0.05],
        'tp':    [0.06, 0.08, 0.10, 0.12],
        'hold':  [1, 2, 3, 5],
        'note':  'Elevated vol, SPY below EMA — very tight, exits-only mode',
    },
}

ALL_REGIMES = list(REGIME_GRIDS.keys())

# ── Bot imports ───────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))

try:
    from auth import authenticate
    from flow_momentum import WEIGHTS as CURRENT_WEIGHTS, get_max_candidates, compute_adx
    from exit_manager import dynamic_take_profit
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

try:
    from data_collector import DEFAULT_UNIVERSE
    UNIVERSE = list(DEFAULT_UNIVERSE)
except Exception:
    UNIVERSE = []

try:
    from pre_market_scanner import SCAN_UNIVERSE
    EXTRA = [s for s in SCAN_UNIVERSE if s not in UNIVERSE]
    ALL_TICKERS = UNIVERSE + EXTRA
except Exception:
    ALL_TICKERS = UNIVERSE

for t in ['SPY', 'QQQ', 'IWM', 'VIXY', 'TLT', 'HYG', 'LQD', 'USO', 'GLD']:
    if t not in ALL_TICKERS:
        ALL_TICKERS.append(t)

# ── Data fetching ─────────────────────────────────────────────────────────────

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


def fetch_price_history(tickers, years=5):
    try:
        import yfinance as yf
    except ImportError:
        print("❌ yfinance not installed")
        sys.exit(1)

    period  = f"{years}y"
    history = {}
    batches = [tickers[i:i+50] for i in range(0, len(tickers), 50)]

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

    # Single-ticker fallback for misses
    for sym in [t for t in tickers if t not in history]:
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


# ── Regime detection (mirrors regime_engine.py) ───────────────────────────────

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


# ── Slippage ──────────────────────────────────────────────────────────────────

_ETF_SYMS  = {'SPY','QQQ','IWM','GLD','TLT','HYG','LQD','USO','GDX','VTI',
               'XLF','XLE','XLK','XLV','VIXY','VXX','UVXY','SQQQ','SPXU'}
_LARGE_CAP = {'AAPL','MSFT','AMZN','GOOGL','META','NVDA','TSLA','JPM',
               'V','MA','UNH','JNJ','XOM','CVX','ABBV','LLY','AVGO'}

def apply_slippage(price, symbol, is_buy):
    if symbol in _ETF_SYMS:    slip = 0.0001
    elif symbol in _LARGE_CAP: slip = 0.0005
    else:                      slip = 0.0010
    return price * (1 + slip) if is_buy else price * (1 - slip)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Pos:
    symbol:      str
    shares:      float
    entry_price: float
    entry_date:  object
    cost_basis:  float
    regime:      str = 'neutral'

@dataclass
class Trade:
    symbol:      str
    pnl:         float
    hold_days:   int
    exit_reason: str
    regime:      str     # regime at EXIT (for regime-split analysis)
    entry_regime: str    # regime at ENTRY


# ── Simplified scoring ────────────────────────────────────────────────────────

def score_symbol(symbol, date, history, spy_hist):
    df = history.get(symbol)
    if df is None or date not in df.index:
        return 0.0

    loc = df.index.get_loc(date)
    if loc < 30:
        return 0.0

    closes  = df['close'].iloc[max(0, loc-60):loc+1]
    volumes = df['volume'].iloc[max(0, loc-60):loc+1] if 'volume' in df.columns else pd.Series()
    sigs    = {}

    # market_tide
    spy = history.get('SPY')
    if spy is not None and date in spy.index:
        sp_loc = spy.index.get_loc(date)
        if sp_loc >= 20:
            sp_c = spy['close'].iloc[sp_loc-20:sp_loc+1]
            ema  = sp_c.ewm(span=20).mean().iloc[-1]
            sigs['market_tide'] = 1.0 if spy['close'].iloc[sp_loc] > ema else 0.0

    # sector_tide (price momentum proxy)
    if len(closes) >= 20:
        ret20 = closes.iloc[-1] / closes.iloc[-20] - 1
        sigs['sector_tide'] = min(1.0, max(0.0, (ret20 + 0.05) / 0.10))

    # price_rvol
    if len(closes) >= 20 and len(volumes) >= 20:
        ret5 = closes.iloc[-1] / closes.iloc[-5] - 1
        avgv = volumes.iloc[-20:-1].mean()
        todv = volumes.iloc[-1]
        rvol = todv / avgv if avgv > 0 else 1.0
        sigs['price_rvol'] = min(1.0, max(0.0, (ret5 * rvol + 0.02) / 0.05))

    # Signals requiring live data — use neutral 0.5
    for s in ['sweep_flow', 'dark_pool', 'politician', 'insider', 'gex', 'etf_flow']:
        sigs[s] = 0.5

    score = sum(CURRENT_WEIGHTS.get(k, 0) * v for k, v in sigs.items())
    return round(score, 4)


# ── Single backtest run ───────────────────────────────────────────────────────

def run_single(history, vix_df, capital,
               stop_pct, tp_pct, hold_days,
               target_regime='all'):
    """
    Run one backtest pass. If target_regime != 'all', only enter trades
    when regime matches, but exits still fire normally.
    Returns dict of per-regime metrics.
    """
    spy_hist = history.get('SPY', pd.DataFrame())
    dates    = sorted(spy_hist.index) if not spy_hist.empty else []
    if not dates:
        return None

    UNIVERSE_LOCAL = [t for t in ALL_TICKERS if t in history]
    positions  = {}
    cash       = capital
    total_fees = 0.0
    trades     = []
    last_regime = 'neutral'

    for date in dates:
        prices = {sym: float(history[sym].loc[date, 'close'])
                  for sym in UNIVERSE_LOCAL
                  if date in history[sym].index}

        if not prices.get('SPY'):
            continue

        vix     = get_vix_on_date(vix_df, date)
        vix_30d = get_vix_30d_avg(vix_df, date)

        # SPY 50d EMA for vol split
        spy_slice = spy_hist[spy_hist.index <= pd.Timestamp(date)].tail(50)
        if len(spy_slice) >= 10:
            spy_price = float(spy_slice['close'].iloc[-1])
            spy_ema50 = float(spy_slice['close'].ewm(span=50, adjust=False).mean().iloc[-1])
        else:
            spy_price = spy_ema50 = 0.0

        regime = detect_regime(vix, vix_30d, spy_price, spy_ema50)

        # ADX gate
        spy_60 = spy_hist[spy_hist.index <= pd.Timestamp(date)].tail(60)
        adx    = compute_adx(spy_60)

        # ── Exits ────────────────────────────────────────────────────────────
        to_exit = []
        for sym, pos in positions.items():
            price = prices.get(sym, 0)
            if price <= 0:
                continue
            pnl_pct = (price - pos.entry_price) / pos.entry_price
            held    = (pd.Timestamp(date) - pd.Timestamp(pos.entry_date)).days

            # Defensive flip → full exit
            if regime == 'volatility-defensive' and last_regime in ('neutral', 'flow', 'volatility-cautious'):
                to_exit.append((sym, price, 'regime_flip_full'))
                continue
            # Cautious flip → half-exit handled implicitly (don't add new positions)
            # Crisis → full exit
            if regime == 'crisis' and last_regime != 'crisis':
                to_exit.append((sym, price, 'regime_flip_full'))
                continue

            if pnl_pct <= -stop_pct:
                to_exit.append((sym, price, 'stop_loss'))
            elif pnl_pct >= tp_pct and pnl_pct > 0:
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
            trades.append(Trade(sym, pnl, held, reason, regime, pos.regime))

        last_regime = regime

        # ── Entries ──────────────────────────────────────────────────────────
        # Skip entries in defensive/crisis, or if not matching target regime
        if regime in ('crisis', 'volatility-defensive'):
            continue
        if target_regime != 'all' and regime != target_regime:
            continue
        if len(positions) >= MAX_POSITIONS:
            continue

        if adx > 0 and adx < 20:
            continue  # ADX gate

        port_val = cash + sum(
            pos.shares * prices.get(s, pos.entry_price)
            for s, pos in positions.items()
        )
        max_new = get_max_candidates(port_val)
        slots   = max_new - len(positions)
        if slots <= 0:
            continue

        # Score candidates
        scores = {}
        for sym in UNIVERSE_LOCAL:
            if sym in positions or sym not in prices:
                continue
            sc = score_symbol(sym, date, history, spy_hist)
            if sc >= MIN_SCORE:
                scores[sym] = sc

        top     = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:slots]
        pos_dol = cash * 0.90 * BASE_POSITION_PCT

        for sym, sc in top:
            price = prices.get(sym, 0)
            if price <= 0 or pos_dol > cash:
                continue
            ep    = apply_slippage(price, sym, True)
            fee   = pos_dol * SPREAD_COST_PCT
            shares = (pos_dol - fee) / ep
            cash  -= pos_dol
            total_fees += fee
            positions[sym] = Pos(sym, shares, ep, date, pos_dol - fee, regime)

    # Liquidate remaining
    final_date = dates[-1]
    for sym, pos in list(positions.items()):
        price = float(history[sym].loc[final_date, 'close']) \
                if final_date in history.get(sym, pd.DataFrame()).index \
                else pos.entry_price
        ep  = apply_slippage(price, sym, False)
        val = pos.shares * ep
        fee = val * SPREAD_COST_PCT
        pnl = (val - fee) - pos.cost_basis
        held = (pd.Timestamp(final_date) - pd.Timestamp(pos.entry_date)).days
        cash += val - fee
        total_fees += fee
        trades.append(Trade(sym, pnl, held, 'liquidation', 'neutral', pos.regime))

    if not trades:
        return None

    # ── Compute per-regime metrics ────────────────────────────────────────────
    def regime_metrics(regime_trades):
        n = len(regime_trades)
        if n == 0:
            return None
        pnls     = [t.pnl for t in regime_trades]
        wins     = [p for p in pnls if p > 0]
        win_rate = len(wins) / n * 100
        total    = sum(pnls)

        # Sharpe from trade P&L series
        s = pd.Series(pnls)
        sharpe = (s.mean() / s.std() * (252 ** 0.5 / 4)) if s.std() > 0 else 0

        # Max drawdown
        peak, maxdd, running = 0, 0.0, 0.0
        for p in pnls:
            running += p
            peak = max(peak, running)
            dd   = (peak - running) / max(peak, 1) * 100
            maxdd = max(maxdd, dd)

        avg_hold = sum(t.hold_days for t in regime_trades) / n
        opp_exits = [t for t in regime_trades
                     if t.exit_reason in ('stop_loss', 'time_stop')]
        opp_pct   = len(opp_exits) / n * 100

        return {
            'sharpe':    round(sharpe, 3),
            'win_rate':  round(win_rate, 1),
            'trades':    n,
            'total_pnl': round(total, 2),
            'maxdd':     round(maxdd, 2),
            'avg_hold':  round(avg_hold, 1),
            'opp_pct':   round(opp_pct, 1),
        }

    # Split trades by entry regime
    by_entry_regime = defaultdict(list)
    for t in trades:
        by_entry_regime[t.entry_regime].append(t)

    all_sells = [t for t in trades if t.exit_reason != 'liquidation']
    overall   = regime_metrics(all_sells) or {}

    result = {'overall': overall}
    for r in ALL_REGIMES:
        m = regime_metrics(by_entry_regime.get(r, []))
        result[r] = m or {}

    result.update(stop=stop_pct, tp=tp_pct, hold=hold_days)
    return result


# ── Per-regime sweep ──────────────────────────────────────────────────────────

def sweep_regime(regime, history, vix_df, capital, top_n, args_years):
    grid   = REGIME_GRIDS[regime]
    combos = list(itertools.product(grid['stop'], grid['tp'], grid['hold']))

    print(f"\n{'='*82}")
    print(f"  REGIME: {regime.upper()}  ({grid['note']})")
    print(f"  {len(combos)} combinations  |  "
          f"Stop: {[f'{s:.0%}' for s in grid['stop']]}  "
          f"TP: {[f'{t:.0%}' for t in grid['tp']]}  "
          f"Hold: {grid['hold']}d")
    print(f"{'='*82}")
    print(f"  {'Stop':>5}  {'TP':>5}  {'Hold':>4}  {'Sharpe':>7}  "
          f"{'WinR':>5}  {'Trades':>6}  {'MaxDD':>6}  {'AvgHold':>7}  "
          f"{'OppCost':>8}  {'TotalPnL':>10}")
    print(f"  {'-'*80}")

    results = []
    t0 = time.time()

    for i, (stop, tp, hold) in enumerate(combos, 1):
        try:
            r = run_single(history, vix_df, capital, stop, tp, hold,
                           target_regime=regime)
            if not r:
                continue
            m = r.get(regime, {})
            if not m or m.get('trades', 0) < 5:
                continue  # skip combos with too few trades to be meaningful

            row = {
                'stop': stop, 'tp': tp, 'hold': hold,
                **m,
            }
            results.append(row)

            best = max((x['sharpe'] for x in results), default=0)
            mark = ' ◀' if row['sharpe'] == best else ''
            eta  = (time.time() - t0) / i * (len(combos) - i)

            print(f"  {stop:>4.0%}  {tp:>4.0%}  {hold:>4}d  "
                  f"{row['sharpe']:>7.3f}  {row['win_rate']:>4.1f}%  "
                  f"{row['trades']:>6}  {row['maxdd']:>5.1f}%  "
                  f"{row['avg_hold']:>6.1f}d  {row['opp_pct']:>7.1f}%  "
                  f"${row['total_pnl']:>9,.0f}{mark}  "
                  f"[ETA {eta/60:.0f}m]")
        except Exception as e:
            print(f"  {stop:.0%}/{tp:.0%}/{hold}d  ERROR: {e}")

    return results


# ── Summary printer ───────────────────────────────────────────────────────────

def print_regime_summary(regime, results, top_n):
    if not results:
        print(f"  ⚠️  No results for {regime} (insufficient trades)")
        return None

    print(f"\n  ── Top {top_n} for {regime.upper()} by Sharpe ──────────────────────────────────")
    print(f"  {'Stop':>5}  {'TP':>5}  {'Hold':>4}  {'Sharpe':>7}  "
          f"{'WinR':>5}  {'Trades':>6}  {'OppCost':>8}")
    print(f"  {'-'*55}")
    top = sorted(results, key=lambda x: x['sharpe'], reverse=True)[:top_n]
    for r in top:
        print(f"  {r['stop']:>4.0%}  {r['tp']:>4.0%}  {r['hold']:>4}d  "
              f"{r['sharpe']:>7.3f}  {r['win_rate']:>4.1f}%  "
              f"{r['trades']:>6}  {r['opp_pct']:>7.1f}%")

    best = top[0]
    return best


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Regime-Aware Exit Parameter Sweep')
    parser.add_argument('--years',   default=5,     type=int,   help='Years of history')
    parser.add_argument('--capital', default=25000, type=float, help='Starting capital')
    parser.add_argument('--top',     default=10,    type=int,   help='Top N per regime')
    parser.add_argument('--regime',  default='all',             help='Regime to sweep (or all)')
    args = parser.parse_args()

    print("🔌 Authenticating with Schwab...")
    client = authenticate()
    print("✅ Connected\n")

    print(f"📥 Fetching {args.years}yr price history ({len(ALL_TICKERS)} tickers)...")
    print("   Data fetched ONCE — all regimes and combos share it\n")
    history = fetch_price_history(ALL_TICKERS, years=args.years)

    print("\n📈 Fetching VIX history from CBOE...")
    vix_df = fetch_vix_history()

    regimes = ALL_REGIMES if args.regime == 'all' else [args.regime]

    all_best = {}
    sweep_start = time.time()

    for regime in regimes:
        if regime not in REGIME_GRIDS:
            print(f"⚠️  Unknown regime: {regime}. Choose from {ALL_REGIMES}")
            continue

        results = sweep_regime(regime, history, vix_df, args.capital, args.top, args.years)
        best    = print_regime_summary(regime, results, args.top)
        if best:
            all_best[regime] = best

    # ── Cross-regime recommendations ─────────────────────────────────────────
    total_elapsed = (time.time() - sweep_start) / 60
    print(f"\n{'='*82}")
    print(f"  REGIME-AWARE EXIT RECOMMENDATIONS")
    print(f"{'='*82}")
    print(f"  {'Regime':<22}  {'Stop':>5}  {'TP':>5}  {'Hold':>4}  "
          f"{'Sharpe':>7}  {'WinR':>5}  {'OppCost':>8}  Note")
    print(f"  {'-'*80}")

    regime_notes = {
        'flow':                  'Let winners run — wide stops ok',
        'neutral':               'Balanced — default settings',
        'volatility-cautious':   'Tighter stops, faster exits',
        'volatility-defensive':  'Survival mode — minimize losses',
    }

    for regime in ALL_REGIMES:
        best = all_best.get(regime)
        if best:
            print(f"  {regime:<22}  {best['stop']:>4.0%}  {best['tp']:>4.0%}  "
                  f"{best['hold']:>4}d  {best['sharpe']:>7.3f}  "
                  f"{best['win_rate']:>4.1f}%  {best['opp_pct']:>7.1f}%  "
                  f"{regime_notes.get(regime,'')}")
        else:
            print(f"  {regime:<22}  {'N/A':>5}  {'N/A':>5}  {'N/A':>4}  "
                  f"{'N/A':>7}  {'N/A':>5}  {'N/A':>8}  insufficient data")

    print(f"\n{'='*82}")
    print(f"  CURRENT SETTINGS (uniform, pre-sweep)")
    print(f"{'='*82}")
    print(f"  {'Regime':<22}  {'Stop':>5}  {'TP':>5}  {'Hold':>4}  Note")
    current = {
        'flow':                 (0.05, 0.22, 5),
        'neutral':              (0.05, 0.15, 5),
        'volatility-cautious':  (0.04, 0.12, 5),
        'volatility-defensive': (0.04, 0.12, 3),
    }
    for regime, (stop, tp, hold) in current.items():
        print(f"  {regime:<22}  {stop:>4.0%}  {tp:>4.0%}  {hold:>4}d")

    print(f"\n✅ Sweep complete — {total_elapsed:.1f} min")
    print(f"   Copy optimal settings into exit_manager.py REGIME_PARAMS dict")


if __name__ == '__main__':
    main()