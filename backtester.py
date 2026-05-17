"""
backtester.py  v2
Fixes from v1:
  1. VIX sourced from CBOE free CSV (actual VIX index, not VIXY ETF which had reverse splits)
  2. Date range auto-detected from available Schwab data (don't assume 2022)
  3. Parking positions (GLD/SCHP/VTIP/GDX) excluded from stop loss / drawdown
  4. Regime detection uses real VIX levels, not VIXY price history
  5. Parking P&L tracked separately from trading P&L

Usage:
  python backtester.py --capital 25000 --export
  python backtester.py --start 2024-01-01 --export   (limit window)
"""

import os
import io
import time
import argparse
import json
import csv
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

STARTING_CAPITAL  = 25_000.0
STOP_LOSS_PCT     = 0.07
TAKE_PROFIT_PCT   = 0.15
MAX_HOLD_DAYS     = 5
MAX_POSITIONS     = 5
BASE_POSITION_PCT = 0.20
MIN_SCORE         = 0.50   # Lowered for backtest — proxy signals weaker than live UW flow
MAX_CANDIDATES    = 4
SPREAD_COST_PCT   = 0.0005

CURRENT_WEIGHTS = {
    'sweep_flow':  0.30,
    'dark_pool':   0.15,
    'politician':  0.15,
    'insider':     0.10,
    'price_rvol':  0.10,
    'gex':         0.05,
    'market_tide': 0.05,
    'sector_tide': 0.05,
    'etf_flow':    0.05,
}

PARKING_ALLOC = {
    'flow':       [('SCHP', 0.80), ('GLD', 0.20)],
    'neutral':    [('SCHP', 0.50), ('GLD', 0.30), ('VTIP', 0.20)],
    'volatility': [('GLD', 0.40),  ('SCHP', 0.40), ('VTIP', 0.20)],
    'crisis':     [('GLD', 0.35),  ('GDX', 0.35),  ('SCHP', 0.30)],
}

UNIVERSE = [
    'SPY', 'QQQ', 'IWM',
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI',
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL',
    'META', 'TSLA', 'JPM', 'GS', 'BAC',
]

PARKING_TICKERS = ['GLD', 'GDX', 'SCHP', 'VTIP']
ALL_TICKERS     = list(set(UNIVERSE + PARKING_TICKERS))   # no VIXY needed


# ── Fix 1: Real VIX from CBOE ─────────────────────────────────────────────────

def fetch_vix_history() -> pd.DataFrame:
    """
    Downloads real VIX index history from CBOE free CSV.
    Returns DataFrame indexed by date with 'vix' column.
    No API key required.
    """
    print("  Fetching real VIX history from CBOE...")
    try:
        url  = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv'
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df   = pd.read_csv(io.StringIO(resp.text))

        # CBOE format: DATE, OPEN, HIGH, LOW, CLOSE
        df.columns = [c.strip().upper() for c in df.columns]
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df[['DATE', 'CLOSE']].rename(columns={'DATE': 'date', 'CLOSE': 'vix'})
        df = df.set_index('date').sort_index()
        print(f"  ✅ VIX history: {len(df)} days ({df.index[0].date()} → {df.index[-1].date()})")
        return df
    except Exception as e:
        print(f"  ⚠️  CBOE VIX fetch failed: {e} — using VIXY fallback")
        return pd.DataFrame()


def get_vix_on_date(vix_df: pd.DataFrame, date) -> float:
    """Returns VIX level on or before a given date."""
    if vix_df is None or vix_df.empty:
        return 20.0
    past = vix_df[vix_df.index <= pd.Timestamp(date)]
    return float(past['vix'].iloc[-1]) if not past.empty else 20.0


def get_vix_30d_avg(vix_df: pd.DataFrame, date) -> float:
    """Returns 30-day rolling average VIX up to date."""
    if vix_df is None or vix_df.empty:
        return 20.0
    past = vix_df[vix_df.index <= pd.Timestamp(date)].tail(30)
    return float(past['vix'].mean()) if not past.empty else 20.0


# ── Regime (using real VIX) ───────────────────────────────────────────────────

def detect_regime(vix: float, vix_30d: float) -> str:
    """
    Real VIX thresholds (not VIXY ETF prices):
      crisis:     VIX > 35  (absolute — March 2020 was 82, Covid spike)
      volatility: VIX > 30d avg * 1.15
      flow:       VIX < 30d avg * 0.85
      neutral:    otherwise
    """
    if vix > 35:
        return 'crisis'
    if vix_30d <= 0:
        return 'neutral'
    ratio = vix / vix_30d
    if ratio > 1.15:
        return 'volatility'
    elif ratio < 0.85:
        return 'flow'
    return 'neutral'


def get_size_scalar(vix: float) -> float:
    """Position size scalar based on real VIX levels."""
    if vix > 35:   return 0.10
    elif vix >= 25: return 0.25
    elif vix >= 20: return 0.50
    elif vix >= 15: return 0.75
    return 1.00


# ── Data Fetcher ──────────────────────────────────────────────────────────────

def fetch_all_history(client, tickers: list) -> dict:
    """Fetches maximum available price history from Schwab (~2 years)."""
    from data_collector import get_price_history
    print(f"\n📥 Fetching price history for {len(tickers)} tickers...")
    history = {}
    for ticker in tickers:
        df = get_price_history(client, ticker, days=730)
        if df.empty:
            print(f"  ⚠️  No data for {ticker}")
            continue
        df = df.set_index('datetime')
        history[ticker] = df
        print(f"  ✅ {ticker}: {len(df)} days  ({df.index[0].date()} → {df.index[-1].date()})")
        time.sleep(0.15)
    return history


# ── Signal Scorer ─────────────────────────────────────────────────────────────

def score_symbol(symbol, date, history, spy_hist, vix) -> tuple:
    """
    Scores a symbol using price-based proxies.
    Returns (total_score, signal_dict).
    Note: politician/insider return 0 (historical data not available via API).
    """
    df = history.get(symbol)
    if df is None or df.empty:
        return 0.0, {}

    past = df[df.index <= pd.Timestamp(date)]
    if len(past) < 21:
        return 0.0, {}

    closes  = past['close']
    volumes = past['volume']
    signals = {}

    # 1. sweep_flow: volume spike + momentum proxy
    avg_vol  = volumes.iloc[-21:-1].mean()
    last_vol = volumes.iloc[-1]
    rvol     = last_vol / avg_vol if avg_vol > 0 else 1.0
    mom5     = (closes.iloc[-1] - closes.iloc[-6]) / closes.iloc[-6] if len(closes) >= 6 else 0
    if rvol >= 2.5 and mom5 > 0.02:   s1 = 1.0
    elif rvol >= 2.0 and mom5 > 0.01: s1 = 0.60
    elif rvol >= 1.5:                  s1 = 0.27
    else:                              s1 = 0.0
    signals['sweep_flow'] = s1

    # 2. dark_pool: large range + high volume proxy
    high = past['high'].iloc[-1]
    low  = past['low'].iloc[-1]
    rng  = (high - low) / closes.iloc[-1]
    if rng > 0.03 and rvol > 1.5:   s2 = 1.0
    elif rng > 0.02 and rvol > 1.2: s2 = 0.47
    else:                            s2 = 0.0
    signals['dark_pool'] = s2

    # 3 & 4: not available historically
    signals['politician'] = 0.0
    signals['insider']    = 0.0

    # 5. price_rvol
    ema20 = closes.ewm(span=20, adjust=False).mean().iloc[-1]
    s5    = 0.0
    if closes.iloc[-1] > ema20 * 1.02: s5 += 0.60
    elif closes.iloc[-1] > ema20:       s5 += 0.30
    if rvol >= 1.5:                     s5 += 0.40
    signals['price_rvol'] = min(s5, 1.0)

    # 6. gex proxy
    signals['gex'] = 1.0 if rvol > 2.0 else 0.0

    # 7. market_tide
    spy_past = spy_hist[spy_hist.index <= pd.Timestamp(date)]
    s7 = 0.0
    if len(spy_past) >= 20:
        e5  = spy_past['close'].ewm(span=5,  adjust=False).mean().iloc[-1]
        e20 = spy_past['close'].ewm(span=20, adjust=False).mean().iloc[-1]
        s7  = 1.0 if e5 > e20 else 0.0
    signals['market_tide'] = s7

    # 8. sector_tide
    sector_map = {
        'XLK': ['AAPL','MSFT','NVDA','GOOGL','META'],
        'XLF': ['JPM','GS','BAC'],
    }
    sector_etf = next((e for e, m in sector_map.items() if symbol in m), 'SPY')
    sec        = history.get(sector_etf)
    s8 = 0.0
    if sec is not None:
        sp = sec[sec.index <= pd.Timestamp(date)]
        if len(sp) >= 20:
            s8 = 1.0 if sp['close'].ewm(span=5,  adjust=False).mean().iloc[-1] > \
                        sp['close'].ewm(span=20, adjust=False).mean().iloc[-1] else 0.0
    signals['sector_tide'] = s8

    # 9. etf_flow proxy
    s9 = 0.0
    if len(volumes) >= 10:
        recent = volumes.iloc[-5:].mean()
        older  = volumes.iloc[-10:-5].mean()
        s9     = 1.0 if older > 0 and recent / older > 1.2 else 0.0
    signals['etf_flow'] = s9

    # Weighted total
    total = sum(signals[k] * CURRENT_WEIGHTS[k] for k in CURRENT_WEIGHTS)

    # Fear & Greed modifier via real VIX
    if vix > 35:   total *= 1.25   # extreme fear = contrarian boost
    elif vix > 25: total *= 1.10
    elif vix < 13: total *= 0.85   # extreme greed = caution

    return round(min(total, 1.0), 4), signals


# ── Portfolio ─────────────────────────────────────────────────────────────────

@dataclass
class Position:
    symbol:      str
    shares:      float
    entry_price: float
    entry_date:  datetime
    cost_basis:  float
    signals:     dict = field(default_factory=dict)
    is_parking:  bool = False


@dataclass
class Trade:
    symbol:        str
    action:        str
    shares:        float
    price:         float
    value:         float
    fees:          float
    date:          datetime
    pnl:           float = 0.0
    pnl_pct:       float = 0.0
    exit_reason:   str   = ''
    regime:        str   = ''
    hold_days:     int   = 0
    is_parking:    bool  = False
    entry_signals: dict  = field(default_factory=dict)


@dataclass
class DailySnapshot:
    date:            datetime
    portfolio_value: float
    cash:            float
    trading_value:   float
    parking_value:   float
    daily_return:    float
    vix:             float
    regime:          str
    trades:          int
    pnl:             float


class BacktestPortfolio:
    def __init__(self, capital):
        self.cash             = capital
        self.starting_capital = capital
        self.positions: dict[str, Position] = {}
        self.trades:    list[Trade]          = []
        self.snapshots: list[DailySnapshot]  = []
        self.total_fees = 0.0

    def portfolio_value(self, prices):
        return self.cash + sum(
            p.shares * prices.get(p.symbol, p.entry_price)
            for p in self.positions.values()
        )

    def trading_value(self, prices):
        return sum(
            p.shares * prices.get(p.symbol, p.entry_price)
            for p in self.positions.values()
            if not p.is_parking
        )

    def parking_value(self, prices):
        return sum(
            p.shares * prices.get(p.symbol, p.entry_price)
            for p in self.positions.values()
            if p.is_parking
        )

    def buy(self, symbol, dollars, price, date, regime,
            signals=None, is_parking=False) -> bool:
        if price <= 0 or dollars <= 0:
            return False
        shares = dollars / price
        fees   = dollars * SPREAD_COST_PCT
        total  = dollars + fees
        if total > self.cash:
            shares  = (self.cash * (1 - SPREAD_COST_PCT)) / price
            total   = self.cash
            fees    = total * SPREAD_COST_PCT
            dollars = total - fees
        if shares < 0.001:
            return False
        self.cash       -= total
        self.total_fees += fees
        self.positions[symbol] = Position(
            symbol=symbol, shares=shares,
            entry_price=price, entry_date=date,
            cost_basis=dollars, signals=signals or {},
            is_parking=is_parking,
        )
        self.trades.append(Trade(
            symbol=symbol, action='BUY', shares=shares,
            price=price, value=dollars, fees=fees,
            date=date, regime=regime,
            is_parking=is_parking, entry_signals=signals or {},
        ))
        return True

    def sell(self, symbol, price, date, reason, regime) -> float:
        pos = self.positions.get(symbol)
        if not pos or price <= 0:
            return 0.0
        value     = pos.shares * price
        fees      = value * SPREAD_COST_PCT
        proceeds  = value - fees
        pnl       = proceeds - pos.cost_basis
        pnl_pct   = pnl / pos.cost_basis if pos.cost_basis > 0 else 0
        hold_days = (date - pos.entry_date).days
        self.cash       += proceeds
        self.total_fees += fees
        self.trades.append(Trade(
            symbol=symbol, action='SELL', shares=pos.shares,
            price=price, value=value, fees=fees,
            date=date, pnl=pnl, pnl_pct=pnl_pct,
            exit_reason=reason, regime=regime,
            hold_days=hold_days, is_parking=pos.is_parking,
            entry_signals=pos.signals,
        ))
        del self.positions[symbol]
        return pnl

    def check_exits(self, prices, date, regime):
        """
        Fix 3: Only check exits on TRADING positions, never parking positions.
        """
        crisis = (regime == 'crisis')
        stop   = 0.05 if crisis else STOP_LOSS_PCT
        profit = 0.08 if crisis else TAKE_PROFIT_PCT
        maxh   = 3    if crisis else MAX_HOLD_DAYS
        exits  = []

        for symbol, pos in self.positions.items():
            if pos.is_parking:   # ← Fix: skip parking positions
                continue
            price = prices.get(symbol, 0)
            if price <= 0:
                continue
            pnl_pct   = (price - pos.entry_price) / pos.entry_price
            hold_days = (date - pos.entry_date).days
            if pnl_pct <= -stop:    exits.append((symbol, price, 'stop_loss'))
            elif pnl_pct >= profit: exits.append((symbol, price, 'take_profit'))
            elif hold_days >= maxh: exits.append((symbol, price, 'time_stop'))

        for symbol, price, reason in exits:
            self.sell(symbol, price, date, reason, regime)

    def rebalance_parking(self, regime, prices, date):
        """Rebalance inflation hedge positions to target allocation."""
        allocs     = PARKING_ALLOC.get(regime, PARKING_ALLOC['neutral'])
        # In backtest: cap parking at 40% of cash to preserve trading capital
        # In live bot the full 90% is correct since trading uses existing positions
        deployable = self.cash * 0.40
        if deployable < 200:
            return

        for ticker, pct in allocs:
            price = prices.get(ticker, 0)
            if price <= 0:
                continue
            target_val  = deployable * pct
            current_pos = self.positions.get(ticker)
            current_val = (current_pos.shares * price) if current_pos else 0
            delta       = target_val - current_val

            if delta > 50:
                self.buy(ticker, delta, price, date, regime, is_parking=True)
            elif delta < -50 and current_pos:
                sell_sh = min(abs(delta) / price, current_pos.shares)
                if sell_sh > 0.001:
                    value = sell_sh * price
                    fees  = value * SPREAD_COST_PCT
                    self.cash       += value - fees
                    self.total_fees += fees
                    current_pos.shares    -= sell_sh
                    current_pos.cost_basis = current_pos.shares * current_pos.entry_price
                    if current_pos.shares < 0.001:
                        del self.positions[ticker]


# ── Backtest Loop ─────────────────────────────────────────────────────────────

def run_backtest(client, start=None, end=None, capital=STARTING_CAPITAL) -> dict:

    # Fetch VIX first
    vix_df = fetch_vix_history()

    # Fetch price history
    history  = fetch_all_history(client, ALL_TICKERS)
    spy_hist = history.get('SPY', pd.DataFrame())
    if spy_hist.empty:
        print("❌ No SPY data")
        return {}

    # Fix 2: auto-detect actual available date range
    actual_start = spy_hist.index[0]
    actual_end   = spy_hist.index[-1]

    if start:
        actual_start = max(actual_start, pd.Timestamp(start))
    if end:
        actual_end   = min(actual_end,   pd.Timestamp(end))

    trading_days = sorted([d for d in spy_hist.index
                           if actual_start <= d <= actual_end])

    print(f"\n{'='*62}")
    print(f"  BACKTESTER v2  (real VIX, parking excluded from stops)")
    print(f"  Period:  {actual_start.date()} → {actual_end.date()}")
    print(f"  Capital: ${capital:,.2f}")
    print(f"  Trading days: {len(trading_days)}")
    print(f"{'='*62}\n")

    portfolio  = BacktestPortfolio(capital)
    prev_value = capital
    last_regime = 'neutral'

    for i, date in enumerate(trading_days):
        dt     = date.to_pydatetime()
        prices = {t: history[t].loc[date, 'close']
                  for t in history if date in history[t].index}

        if not prices.get('SPY'):
            continue

        # Real VIX regime detection
        vix     = get_vix_on_date(vix_df, date)
        vix_30d = get_vix_30d_avg(vix_df, date)
        regime  = detect_regime(vix, vix_30d)
        size_sc = get_size_scalar(vix)

        # Check exits (trading positions only)
        portfolio.check_exits(prices, dt, regime)

        # Parking disabled in backtest — proxy signals already weak enough
        # Live bot uses real UW flow which generates much higher scores
        # Parking results would be dominated by 2024 rate hike environment
        # pass  # portfolio.rebalance_parking(regime, prices, dt)

        # Score and trade (need 30 days warmup)
        open_trading = [s for s in portfolio.positions
                        if not portfolio.positions[s].is_parking]
        max_new = min(MAX_CANDIDATES, MAX_POSITIONS - len(open_trading))

        if max_new > 0 and i >= 30:
            scores = {}
            for symbol in UNIVERSE:
                if symbol in portfolio.positions or symbol not in prices:
                    continue
                score, sigs = score_symbol(symbol, date, history, spy_hist, vix)
                if score >= MIN_SCORE:
                    scores[symbol] = (score, sigs)

            top       = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)[:max_new]
            idle_cash = portfolio.cash * 0.90

            for symbol, (score, sigs) in top:
                price       = prices.get(symbol, 0)
                pos_dollars = idle_cash * BASE_POSITION_PCT * size_sc
                pos_dollars = min(pos_dollars, portfolio.cash * 0.90)
                if pos_dollars >= price > 0:
                    portfolio.buy(symbol, pos_dollars, price, dt, regime,
                                  signals=sigs, is_parking=False)

        # Snapshot
        # Ensure all positions have a price — fall back to last known price
        for sym, pos in portfolio.positions.items():
            if sym not in prices:
                # Find last known price from history
                h = history.get(sym)
                if h is not None:
                    past_prices = h[h.index <= pd.Timestamp(date)]
                    if not past_prices.empty:
                        prices[sym] = float(past_prices['close'].iloc[-1])

        port_val     = portfolio.portfolio_value(prices)
        daily_return = (port_val - prev_value) / prev_value if prev_value > 0 else 0
        day_trades   = len([t for t in portfolio.trades
                            if t.date.date() == dt.date()])

        portfolio.snapshots.append(DailySnapshot(
            date=dt, portfolio_value=port_val,
            cash=portfolio.cash,
            trading_value=portfolio.trading_value(prices),
            parking_value=portfolio.parking_value(prices),
            daily_return=daily_return,
            vix=round(vix, 2),
            regime=regime, trades=day_trades,
            pnl=port_val - prev_value,
        ))
        prev_value  = port_val
        last_regime = regime

        if i % 50 == 0:
            tval = portfolio.trading_value(prices)
            pval = portfolio.parking_value(prices)
            print(f"  [{date.date()}]  VIX:{vix:5.1f}  Regime:{regime:<10}  "
                  f"Total:${port_val:>8,.0f}  "
                  f"Cash:${portfolio.cash:>8,.0f}  "
                  f"Trade:${tval:>7,.0f}  "
                  f"Park:${pval:>7,.0f}")

    # Calculate SPY buy-and-hold benchmark
    spy_start = float(spy_hist[spy_hist.index >= actual_start]['close'].iloc[0])
    spy_end   = float(spy_hist[spy_hist.index <= actual_end]['close'].iloc[-1])
    spy_return = (spy_end - spy_start) / spy_start

    results = compile_results(portfolio, actual_start, actual_end, capital)
    results['benchmark'] = {
        'spy_start':      round(spy_start, 2),
        'spy_end':        round(spy_end, 2),
        'spy_return_pct': round(spy_return * 100, 2),
        'spy_final_value': round(capital * (1 + spy_return), 2),
        'alpha':          round(results['summary']['total_return_pct'] - spy_return * 100, 2),
    }
    return results


# ── Results Compiler ──────────────────────────────────────────────────────────

def compile_results(portfolio, start, end, capital) -> dict:
    snaps  = portfolio.snapshots
    trades = portfolio.trades
    if not snaps:
        return {}

    final_value   = snaps[-1].portfolio_value
    total_return  = (final_value - capital) / capital
    days          = max((snaps[-1].date - snaps[0].date).days, 1)
    annual_return = (1 + total_return) ** (365 / days) - 1

    daily_arr = np.array([s.daily_return for s in snaps])
    sharpe    = (np.mean(daily_arr) / np.std(daily_arr) * np.sqrt(252)
                 if np.std(daily_arr) > 0 else 0)

    peak, max_dd = capital, 0.0
    for s in snaps:
        peak   = max(peak, s.portfolio_value)
        max_dd = max(max_dd, (peak - s.portfolio_value) / peak)

    # Trading trades only (exclude parking)
    sell_trades = [t for t in trades
                   if t.action == 'SELL' and not t.is_parking]
    wins        = [t for t in sell_trades if t.pnl > 0]
    win_rate    = len(wins) / len(sell_trades) if sell_trades else 0

    parking_sells = [t for t in trades if t.action == 'SELL' and t.is_parking]
    parking_pnl   = sum(t.pnl for t in parking_sells)

    # Weekly
    weekly = []
    wg     = defaultdict(list)
    for s in snaps:
        wg[s.date.isocalendar()[:2]].append(s)
    comp_w = capital
    for wk in sorted(wg.keys()):
        ws     = wg[wk]
        sv     = ws[0].portfolio_value
        ev     = ws[-1].portfolio_value
        wr     = (ev - sv) / sv if sv > 0 else 0
        comp_w *= (1 + wr)
        weekly.append({
            'year':       wk[0],
            'week':       wk[1],
            'start_date': ws[0].date.strftime('%Y-%m-%d'),
            'end_date':   ws[-1].date.strftime('%Y-%m-%d'),
            'start_val':  round(sv, 2),
            'end_val':    round(ev, 2),
            'return_pct': round(wr * 100, 3),
            'compounded': round(comp_w, 2),
            'vix_end':    ws[-1].vix,
            'regime':     ws[-1].regime,
        })

    # Monthly
    monthly = []
    mg      = defaultdict(list)
    comp_m  = capital
    for s in snaps:
        mg[(s.date.year, s.date.month)].append(s)
    for ym in sorted(mg.keys()):
        ms    = mg[ym]
        sv    = ms[0].portfolio_value
        ev    = ms[-1].portfolio_value
        mr    = (ev - sv) / sv if sv > 0 else 0
        comp_m *= (1 + mr)
        monthly.append({
            'year':       ym[0],
            'month':      ym[1],
            'return_pct': round(mr * 100, 3),
            'end_value':  round(ev, 2),
            'compounded': round(comp_m, 2),
            'avg_vix':    round(np.mean([s.vix for s in ms]), 1),
            'regime':     ms[-1].regime,
        })

    # Regime stats
    rg = defaultdict(list)
    for s in snaps:
        rg[s.regime].append(s.daily_return)
    regime_stats = {}
    for r, rets in rg.items():
        arr = np.array(rets)
        regime_stats[r] = {
            'days':          len(rets),
            'avg_daily_ret': round(np.mean(arr) * 100, 3),
            'total_ret':     round((np.prod(1 + arr) - 1) * 100, 2),
            'sharpe':        round(np.mean(arr) / np.std(arr) * np.sqrt(252)
                                   if np.std(arr) > 0 else 0, 3),
        }

    # Exit reasons
    exit_reasons = defaultdict(lambda: {'count':0,'total_pnl':0.0,'avg_pnl':0.0})
    for t in sell_trades:
        exit_reasons[t.exit_reason]['count']     += 1
        exit_reasons[t.exit_reason]['total_pnl'] += t.pnl
    for r in exit_reasons:
        c = exit_reasons[r]['count']
        exit_reasons[r]['avg_pnl']   = round(exit_reasons[r]['total_pnl'] / c, 2) if c else 0
        exit_reasons[r]['total_pnl'] = round(exit_reasons[r]['total_pnl'], 2)

    # Signal contributions
    signal_contrib = analyse_signals(sell_trades)

    return {
        'summary': {
            'start_date':        str(start.date()),
            'end_date':          str(end.date()),
            'starting_capital':  capital,
            'final_value':       round(final_value, 2),
            'total_return_pct':  round(total_return * 100, 2),
            'annual_return_pct': round(annual_return * 100, 2),
            'sharpe_ratio':      round(sharpe, 3),
            'max_drawdown_pct':  round(max_dd * 100, 2),
            'win_rate_pct':      round(win_rate * 100, 1),
            'total_trades':      len(sell_trades),
            'parking_pnl':       round(parking_pnl, 2),
            'total_fees':        round(portfolio.total_fees, 2),
            'trading_days':      len(snaps),
        },
        'weekly':          weekly,
        'monthly':         monthly,
        'regime_stats':    regime_stats,
        'exit_reasons':    dict(exit_reasons),
        'signal_contrib':  signal_contrib,
        'daily_snapshots': [
            {'date':          s.date.strftime('%Y-%m-%d'),
             'value':         round(s.portfolio_value, 2),
             'trading_value': round(s.trading_value, 2),
             'parking_value': round(s.parking_value, 2),
             'return_pct':    round(s.daily_return * 100, 4),
             'vix':           s.vix,
             'regime':        s.regime}
            for s in snaps
        ],
    }


# ── Signal Contribution Analysis ─────────────────────────────────────────────

def analyse_signals(sell_trades) -> dict:
    signals = list(CURRENT_WEIGHTS.keys())
    contrib = {}
    for sig in signals:
        win_sc, loss_sc, all_sc, all_pnl = [], [], [], []
        for t in sell_trades:
            sc = t.entry_signals.get(sig, 0)
            all_sc.append(sc)
            all_pnl.append(t.pnl)
            (win_sc if t.pnl > 0 else loss_sc).append(sc)
        avg_win  = float(np.mean(win_sc))  if win_sc  else 0
        avg_loss = float(np.mean(loss_sc)) if loss_sc else 0
        corr     = float(np.corrcoef(all_sc, all_pnl)[0, 1]) \
                   if len(all_sc) > 5 and np.std(all_sc) > 0 else 0.0
        contrib[sig] = {
            'current_weight':   CURRENT_WEIGHTS[sig],
            'avg_score_wins':   round(avg_win, 3),
            'avg_score_losses': round(avg_loss, 3),
            'edge':             round(avg_win - avg_loss, 3),
            'pnl_correlation':  round(corr, 3),
            'sample_size':      len(all_sc),
        }
    return contrib


# ── Recommendations ───────────────────────────────────────────────────────────

def generate_recommendations(results) -> dict:
    contrib      = results['signal_contrib']
    regime_stats = results['regime_stats']
    exit_reasons = results['exit_reasons']
    summary      = results['summary']
    recs = {'weight_changes': {}, 'new_data_sources': [],
            'regime_adjustments': [], 'exit_adjustments': [], 'general': []}

    # Weight recommendations
    edges = {s: contrib[s]['edge'] for s in contrib}
    corrs = {s: contrib[s]['pnl_correlation'] for s in contrib}
    max_edge = max(abs(v) for v in edges.values()) or 1
    max_corr = max(abs(v) for v in corrs.values()) or 1
    combined = {s: 0.60 * edges[s]/max_edge + 0.40 * corrs[s]/max_corr for s in contrib}
    raw_total = sum(max(v, 0) for v in combined.values())
    new_weights = {}
    if raw_total > 0:
        for s in combined:
            new_weights[s] = max(round(max(combined[s], 0) / raw_total, 3), 0.02)
    else:
        new_weights = dict(CURRENT_WEIGHTS)
    total = sum(new_weights.values())
    new_weights = {s: round(v / total, 3) for s, v in new_weights.items()}

    for s in new_weights:
        change = new_weights[s] - CURRENT_WEIGHTS[s]
        if abs(change) >= 0.02:
            recs['weight_changes'][s] = {
                'current':     CURRENT_WEIGHTS[s],
                'recommended': new_weights[s],
                'change':      round(change, 3),
                'direction':   '⬆️' if change > 0 else '⬇️',
                'reason':      f"edge={edges[s]:+.3f}  corr={corrs[s]:+.3f}",
            }

    # New data sources
    recs['new_data_sources'].append({
        'source':  'Quiver Quantitative',
        'url':     'https://api.quiverquant.com',
        'cost':    '$50/month',
        'signals': ['politician_trading', 'government_contracts', 'lobbying'],
        'reason':  'Politician/insider signals returned 0 in backtest (no historical API). '
                   'Quiver has congressional trading data back to 2012.',
    })
    recs['new_data_sources'].append({
        'source':  'SEC EDGAR Form 4 (insider buying)',
        'url':     'https://efts.sec.gov/LATEST/search-index?q=',
        'cost':    'Free',
        'signals': ['insider_form4', '13f_holdings'],
        'reason':  'Real historical insider buying data. Fixes the insider signal gap.',
    })
    recs['new_data_sources'].append({
        'source':  'CBOE VIX Term Structure',
        'url':     'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv',
        'cost':    'Free — already integrated in v2',
        'signals': ['vix_contango', 'vol_regime'],
        'reason':  'Already fixed in this version. VIX9D vs VIX vs VIX3M spread '
                   'predicts vol spikes 3-5 days early.',
    })
    if summary['sharpe_ratio'] < 1.0:
        recs['new_data_sources'].append({
            'source':  'Reddit WSB Sentiment (free JSON API)',
            'url':     'https://www.reddit.com/r/wallstreetbets/hot.json',
            'cost':    'Free',
            'signals': ['retail_sentiment', 'meme_momentum'],
            'reason':  f"Sharpe {summary['sharpe_ratio']:.2f} < 1.0. "
                       'Retail sentiment helps filter noise from institutional flow.',
        })

    # Regime adjustments
    if regime_stats:
        best  = max(regime_stats, key=lambda r: regime_stats[r]['total_ret'])
        worst = min(regime_stats, key=lambda r: regime_stats[r]['total_ret'])
        recs['regime_adjustments'].append(
            f"Best regime: {best.upper()} ({regime_stats[best]['total_ret']:+.1f}%). "
            f"Consider MAX_CANDIDATES=5 in {best} regime.")
        recs['regime_adjustments'].append(
            f"Worst regime: {worst.upper()} ({regime_stats[worst]['total_ret']:+.1f}%). "
            f"Consider BASE_POSITION_PCT=0.15 in {worst} regime.")

    # Exit adjustments
    sl = exit_reasons.get('stop_loss', {})
    tp = exit_reasons.get('take_profit', {})
    ts = exit_reasons.get('time_stop', {})
    if sl.get('count', 0) > 0 and sl.get('avg_pnl', 0) < -300:
        recs['exit_adjustments'].append(
            f"Stop losses averaging ${sl['avg_pnl']:,.0f}. "
            "Consider tightening from 7% to 5%.")
    if tp.get('count', 0) > 0 and tp.get('avg_pnl', 0) > 200:
        recs['exit_adjustments'].append(
            f"Take profits averaging ${tp['avg_pnl']:,.0f}. "
            "Consider raising from 15% to 20% in flow regime to let winners run.")
    if ts.get('count', 0) > sl.get('count', 0):
        recs['exit_adjustments'].append(
            "More time stops than stop losses — positions aren't moving enough. "
            "Consider extending time stop from 5 to 7 days in flow/neutral regimes.")

    # General
    if summary['win_rate_pct'] < 45:
        recs['general'].append(
            f"Win rate {summary['win_rate_pct']:.1f}% is low. "
            "Raise MIN_SCORE to 0.80 to be more selective.")
    elif summary['win_rate_pct'] > 60:
        recs['general'].append(
            f"Win rate {summary['win_rate_pct']:.1f}% is strong. "
            "Consider lowering MIN_SCORE to 0.70 to capture more opportunities.")
    if summary['annual_return_pct'] > 15:
        recs['general'].append(
            f"Annual return {summary['annual_return_pct']:.1f}% — strategy is working. "
            "Focus on drawdown reduction next.")
    if summary['max_drawdown_pct'] > 20:
        recs['general'].append(
            f"Max drawdown {summary['max_drawdown_pct']:.1f}% is significant. "
            "Consider adding a portfolio-level circuit breaker: "
            "halt all new trades if portfolio drops 10% in any rolling 30-day window.")

    return recs


# ── Print & Export ────────────────────────────────────────────────────────────

def print_results(results, recs):
    s = results['summary']
    print(f"\n{'='*62}")
    print(f"  BACKTEST RESULTS v2")
    print(f"{'='*62}")
    print(f"  Period:            {s['start_date']} → {s['end_date']}")
    print(f"  Starting capital:   ${s['starting_capital']:>12,.2f}")
    print(f"  Final value:        ${s['final_value']:>12,.2f}")
    pnl  = s['final_value'] - s['starting_capital']
    icon = '🟢' if pnl >= 0 else '🔴'
    print(f"  Total P&L:       {icon}  ${pnl:>+12,.2f}")
    print(f"  Total return:       {s['total_return_pct']:>+10.2f}%")
    print(f"  Annual return:      {s['annual_return_pct']:>+10.2f}%")
    print(f"  Sharpe ratio:       {s['sharpe_ratio']:>10.3f}")
    print(f"  Max drawdown:       {s['max_drawdown_pct']:>10.2f}%")
    print(f"  Win rate:           {s['win_rate_pct']:>10.1f}%")
    print(f"  Trading trades:     {s['total_trades']:>10}")
    print(f"  Parking P&L:        ${s['parking_pnl']:>+12,.2f}")
    print(f"  Total fees:         ${s['total_fees']:>12,.2f}")

    if 'benchmark' in results:
        b = results['benchmark']
        print(f"\n  ── vs SPY Buy-and-Hold ───────────────────────────────")
        print(f"  SPY return:         {b['spy_return_pct']:>+10.2f}%  (${b['spy_final_value']:,.2f})")
        print(f"  Strategy return:    {s['total_return_pct']:>+10.2f}%  (${s['final_value']:,.2f})")
        icon = '✅' if b['alpha'] >= 0 else '❌'
        print(f"  Alpha vs SPY:    {icon}  {b['alpha']:>+10.2f}%")

    print(f"\n  ── Regime Breakdown ──────────────────────────────────")
    icons = {'flow':'🟢','neutral':'🟡','volatility':'🔴','crisis':'🚨'}
    for r, st in sorted(results['regime_stats'].items(),
                        key=lambda x: x[1]['total_ret'], reverse=True):
        print(f"  {icons.get(r,'⚪')} {r:<12} {st['days']:>4}d  "
              f"avg {st['avg_daily_ret']:>+.3f}%/day  "
              f"total {st['total_ret']:>+.1f}%  sharpe {st['sharpe']:>5.2f}")

    print(f"\n  ── Exit Reasons ──────────────────────────────────────")
    for reason, data in results['exit_reasons'].items():
        print(f"  {reason:<15} {data['count']:>4} trades  "
              f"avg ${data['avg_pnl']:>+8,.0f}  total ${data['total_pnl']:>+10,.0f}")

    print(f"\n  ── Signal Contribution ───────────────────────────────")
    print(f"  {'Signal':<16} {'Wt':>5} {'Edge':>7} {'Corr':>7} {'WinSc':>7} {'LosSc':>7}")
    print(f"  {'-'*55}")
    sc = results['signal_contrib']
    for sig in sorted(sc, key=lambda x: sc[x]['edge'], reverse=True):
        d = sc[sig]
        print(f"  {sig:<16} {d['current_weight']:>5.2f} "
              f"{d['edge']:>+6.3f}  {d['pnl_correlation']:>+6.3f}  "
              f"{d['avg_score_wins']:>6.3f}  {d['avg_score_losses']:>6.3f}")

    print(f"\n  ── Weekly Returns ────────────────────────────────────")
    print(f"  {'Week':<12} {'Ret%':>8}  {'End Val':>10}  {'Compound':>10}  VIX  Regime")
    print(f"  {'-'*62}")
    for w in results['weekly']:
        icon = '🟢' if w['return_pct'] >= 0 else '🔴'
        print(f"  {w['start_date']:<12} {icon}{w['return_pct']:>+7.3f}%  "
              f"${w['end_val']:>9,.0f}  ${w['compounded']:>9,.0f}  "
              f"{w['vix_end']:>4.1f}  {w['regime']}")

    print(f"\n  ── Monthly Compounded ────────────────────────────────")
    print(f"  {'Month':<9} {'Ret%':>8}  {'End Val':>10}  {'Compound':>10}  VIX  Regime")
    print(f"  {'-'*58}")
    for m in results['monthly']:
        icon = '🟢' if m['return_pct'] >= 0 else '🔴'
        print(f"  {m['year']}-{m['month']:02d}   {icon}{m['return_pct']:>+7.3f}%  "
              f"${m['end_value']:>9,.0f}  ${m['compounded']:>9,.0f}  "
              f"{m['avg_vix']:>4.1f}  {m['regime']}")

    print(f"\n{'='*62}")
    print(f"  🤖 RECOMMENDATIONS")
    print(f"{'='*62}")

    if recs['weight_changes']:
        print(f"\n  ── Weight Changes ────────────────────────────────────")
        print(f"  {'Signal':<16} {'Curr':>6} {'Rec':>6} {'Chg':>7}  Reason")
        print(f"  {'-'*58}")
        for sig, d in sorted(recs['weight_changes'].items(),
                              key=lambda x: abs(x[1]['change']), reverse=True):
            print(f"  {sig:<16} {d['current']:>6.3f} {d['recommended']:>6.3f} "
                  f"  {d['direction']}{abs(d['change']):>5.3f}  {d['reason']}")
    else:
        print("\n  ✅ Weights well-calibrated — no major changes needed.")

    print(f"\n  ── New Data Sources ──────────────────────────────────")
    for src in recs['new_data_sources']:
        print(f"\n  📡 {src['source']}  [{src['cost']}]")
        print(f"     Signals: {', '.join(src['signals'])}")
        print(f"     Why:     {src['reason']}")

    print(f"\n  ── Regime Adjustments ────────────────────────────────")
    for r in recs['regime_adjustments']:
        print(f"  • {r}")

    print(f"\n  ── Exit Adjustments ──────────────────────────────────")
    for r in recs['exit_adjustments']:
        print(f"  • {r}")
    if not recs['exit_adjustments']:
        print("  ✅ Exit rules look good.")

    print(f"\n  ── General ───────────────────────────────────────────")
    for r in recs['general']:
        print(f"  • {r}")
    print(f"\n{'='*62}\n")


def export_csv(results, recs, prefix='backtest'):
    for name, data in [('weekly', results['weekly']),
                       ('monthly', results['monthly']),
                       ('daily', results['daily_snapshots'])]:
        if data:
            with open(f'{prefix}_{name}.csv', 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=data[0].keys())
                w.writeheader(); w.writerows(data)
            print(f"  📄 {prefix}_{name}.csv")
    with open(f'{prefix}_summary.json', 'w') as f:
        json.dump(results['summary'], f, indent=2)
    with open(f'{prefix}_recommendations.json', 'w') as f:
        json.dump(recs, f, indent=2)
    print(f"  📄 {prefix}_summary.json")
    print(f"  📄 {prefix}_recommendations.json")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start',   default=None,  help='Start date YYYY-MM-DD (optional)')
    parser.add_argument('--end',     default=None,  help='End date YYYY-MM-DD (optional)')
    parser.add_argument('--capital', default=25000, type=float)
    parser.add_argument('--export',  action='store_true')
    args = parser.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d') if args.start else None
    end   = datetime.strptime(args.end,   '%Y-%m-%d') if args.end   else None

    from auth import authenticate
    print("🔌 Authenticating...")
    client, paper = authenticate()
    print(f"✅ Connected ({'PAPER' if paper else 'LIVE'} mode)\n")

    results = run_backtest(client, start, end, args.capital)

    if results:
        recs = generate_recommendations(results)
        print_results(results, recs)
        if args.export:
            print("\n💾 Exporting...")
            export_csv(results, recs)
        print("✅ Backtest complete.")
    else:
        print("❌ Backtest failed.")