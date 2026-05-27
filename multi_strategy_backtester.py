"""
multi_strategy_backtester.py
-----------------------------------------------------------------------------
Tests 4 options strategies across ALL tickers in the DB that have both puts
and calls, over a fixed 1-year window (apples-to-apples comparison).

Strategies tested in parallel (each gets $25k starting capital PER TICKER):
  1. WHEEL       -- CSP -> assignment -> CC -> called away (full cycle)
  2. PURE_CSP    -- Sell 8% OTM puts, close at 50% profit OR liquidate on assignment
  3. LONG_CALL   -- Buy 8% OTM calls, hold to expiry, value at intrinsic
  4. BUY_HOLD    -- Benchmark, buy underlying day 1 and hold

Every trade is tagged with the regime it was ENTERED in.
The regime classifier is the same one used by the live bot's backtester.

Output:
  - strategy_regime_matrix.csv  : core deliverable (strategy x regime metrics)
  - all_trades.csv              : every trade across all strategies
  - equity_curves.csv           : daily portfolio values
  - regime_timeline.csv         : daily regime classification
  - multi_strategy_summary.json : top-line metrics

Usage:
  python multi_strategy_backtester.py
"""

import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
import math

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    print("[X] yfinance not installed. Run: pip install yfinance pandas numpy")
    sys.exit(1)


# -- Config -----------------------------------------------------------------

DB_PATH          = r'C:\trading-bot\data\options_history.db'
SCORES_CSV       = r'C:\trading-bot\backtest_daily_scores.csv'
# Minimum starting capital. Each ticker gets scaled UP from this if needed
# so that 1 contract of the underlying is affordable on day 1.
# e.g. if first put strike is $370, ticker gets ceil(370*100*1.10/5000)*5000 = $45k
MIN_STARTING_CAPITAL = 25_000.0
WHEEL_STOP_PCT   = 0.10
CSP_PROFIT_TAKE  = 0.50    # close pure-CSP at 50% of max profit
CSP_DTE_CLOSE    = 5       # close pure-CSP if ITM and DTE < 5
CALL_PROFIT_TAKE = 1.00    # 100% gain on long call (unused in current build)
CALL_STOP_LOSS   = 0.50    # 50% loss on long call (unused in current build)
RISK_FREE_RATE   = 0.04
TRADING_DAYS     = 252

# Force ALL tickers to the same 1-year window for apples-to-apples comparison.
# AAPL has 5 years available but we cap it here so cross-ticker results are fair.
ANALYSIS_START   = '2025-05-21'
ANALYSIS_END     = '2026-05-21'

# Score bucketing (calibrated to actual distribution: median ~0.52, p75 ~0.61)
SCORE_BUCKETS = [
    ('LOW',    -0.01, 0.50),   # below MIN_SCORE  -- live bot would skip
    ('MEDIUM',  0.50, 0.70),   # above min, normal conviction
    ('HIGH',    0.70, 1.01),   # strong conviction (CSP threshold 0.85 sits here)
]


def bucket_score(score) -> str:
    """Map a score to its bucket name. Returns 'NO_SCORE' if NaN/missing."""
    if score is None or (isinstance(score, float) and score != score):
        return 'NO_SCORE'
    for name, lo, hi in SCORE_BUCKETS:
        if lo <= score < hi:
            return name
    return 'NO_SCORE'


# -- Regime classifier (copied from backtester.py, line 347) ----------------

def detect_regime(vix: float, vix_30d: float,
                  spy_price: float = 0.0, spy_ema50: float = 0.0) -> str:
    """
    Same rules as the live backtester:
      crisis:               VIX > 35
      volatility-defensive: ratio > 1.30 OR SPY below 50d EMA
      volatility-cautious:  ratio 1.15-1.30 AND SPY above 50d EMA
      flow:                 ratio < 0.85
      neutral:              otherwise
    """
    if vix > 35:
        return 'crisis'
    if vix_30d <= 0:
        return 'neutral'
    ratio = vix / vix_30d
    if ratio > 1.15:
        spy_below_ema = (spy_price > 0 and spy_ema50 > 0 and spy_price < spy_ema50)
        if ratio > 1.30 or spy_below_ema:
            return 'volatility-defensive'
        return 'volatility-cautious'
    elif ratio < 0.85:
        return 'flow'
    return 'neutral'


def build_regime_timeline(start: str, end: str) -> pd.DataFrame:
    """Pulls VIX + SPY from yfinance and classifies each day's regime."""
    print(f"   Loading VIX + SPY for regime classification...")
    vix = yf.download('^VIX', start=start, end=end, progress=False, auto_adjust=False)
    spy = yf.download('SPY', start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    vix_close = vix['Close'].astype(float)
    spy_close = spy['Close'].astype(float)

    vix_30d   = vix_close.rolling(30).mean()
    spy_ema50 = spy_close.ewm(span=50, adjust=False).mean()

    df = pd.DataFrame({
        'vix':       vix_close,
        'vix_30d':   vix_30d,
        'spy':       spy_close,
        'spy_ema50': spy_ema50,
    }).dropna()

    df['regime'] = df.apply(
        lambda r: detect_regime(r['vix'], r['vix_30d'], r['spy'], r['spy_ema50']),
        axis=1
    )
    df.index = df.index.strftime('%Y-%m-%d')
    return df


# -- Data loaders -----------------------------------------------------------

def load_options(db_path: str, ticker: str) -> dict:
    """Load all puts and calls for ticker. Returns {'put':{date:row}, 'call':{date:row}}."""
    if not os.path.exists(db_path):
        print(f"[X] DB not found: {db_path}")
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    rows = conn.execute('''
        SELECT side, trade_date, expiration, strike, bid, ask, mid,
               underlying_price, dte
        FROM options_history
        WHERE symbol = ?
        ORDER BY trade_date, dte
    ''', (ticker,)).fetchall()
    conn.close()

    out = {'put': {}, 'call': {}}
    for side, tdate, exp, strike, bid, ask, mid, under, dte in rows:
        if side not in out:
            continue
        if tdate in out[side]:
            existing = out[side][tdate]
            # Prefer the option closest to 45 DTE
            if abs((existing.get('dte') or 99) - 45) <= abs((dte or 99) - 45):
                continue
        out[side][tdate] = {
            'expiration': exp,
            'strike': float(strike),
            'bid': bid, 'ask': ask, 'mid': mid,
            'underlying_price': under,
            'dte': dte,
        }
    return out


def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    df.index = df.index.strftime('%Y-%m-%d')
    return df


def load_scores(csv_path: str) -> dict:
    """Load backtest_daily_scores.csv into a {(ticker, date): score} dict.
    Returns empty dict if file missing -- backtester runs without score tagging."""
    if not os.path.exists(csv_path):
        print(f"   [warn] Score CSV not found at {csv_path}")
        print(f"          Trades will not be tagged with scores.")
        return {}
    df = pd.read_csv(csv_path, usecols=['date', 'ticker', 'score'])
    print(f"   Loaded {len(df):,} score rows from {csv_path}")
    # Build dict for fast lookup
    return {(row.ticker, row.date): row.score for row in df.itertuples(index=False)}


# -- Trade record (shared across strategies) --------------------------------

@dataclass
class Trade:
    strategy: str
    trade_id: int
    entry_date: str
    entry_regime: str
    entry_price: float
    leg_type: str                 # 'short_put'|'short_call'|'long_call'|'long_stock'
    strike: Optional[float] = None
    expiry: Optional[str] = None
    contracts: int = 1
    exit_date: Optional[str] = None
    exit_regime: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    days_held: int = 0
    entry_score: Optional[float] = None       # composite score on entry date
    entry_score_bucket: str = 'NO_SCORE'      # bucketed for matrix analysis


# -- Base strategy class ----------------------------------------------------

class Strategy:
    name = 'BASE'

    def __init__(self, starting_cash: float):
        self.cash = starting_cash
        self.trades: list = []
        self.equity_curve: list = []   # [(date, value)]
        self.trade_counter = 0

    def portfolio_value(self, close_price: float) -> float:
        raise NotImplementedError

    def step(self, dt, close, regime, options):
        raise NotImplementedError

    def record_equity(self, dt, close):
        self.equity_curve.append({
            'date': dt,
            'strategy': self.name,
            'value': self.portfolio_value(close),
        })


# -- Strategy 1: WHEEL ------------------------------------------------------

class WheelStrategy(Strategy):
    name = 'WHEEL'

    def __init__(self, starting_cash):
        super().__init__(starting_cash)
        self.state = 'CASH'   # CASH | CSP_OPEN | HOLDING | HOLDING_CC
        self.shares = 0
        self.cost_basis = 0.0
        self.open_put = None
        self.open_call = None
        self.cur_trade_put = None
        self.cur_trade_call = None

    def portfolio_value(self, close_price):
        return self.cash + self.shares * close_price

    def step(self, dt, close, regime, options):
        # 1. Wheel stop check
        if self.shares > 0 and self.cost_basis > 0 and close < self.cost_basis * (1 - WHEEL_STOP_PCT):
            self._wheel_stop(dt, close, regime)

        # 2. Handle CSP expiry
        if self.state == 'CSP_OPEN' and dt >= self.open_put['expiry']:
            self._handle_csp_expiry(dt, close, regime)

        # 3. Handle CC expiry
        if self.state == 'HOLDING_CC' and dt >= self.open_call['expiry']:
            self._handle_cc_expiry(dt, close, regime)

        # 4. Open new positions
        if self.state == 'CASH':
            self._try_open_csp(dt, close, regime, options)
        elif self.state == 'HOLDING':
            self._try_open_cc(dt, close, regime, options)

    def _try_open_csp(self, dt, close, regime, options):
        if dt not in options['put']: return
        opt = options['put'][dt]
        if not opt['mid'] or opt['mid'] <= 0: return
        if self.cash < opt['strike'] * 100: return

        prem = opt['mid'] * 100
        self.cash += prem
        self.trade_counter += 1
        self.cur_trade_put = Trade(
            strategy=self.name, trade_id=self.trade_counter,
            entry_date=dt, entry_regime=regime, entry_price=opt['mid'],
            leg_type='short_put', strike=opt['strike'], expiry=opt['expiration'],
        )
        self.open_put = {
            'strike': opt['strike'], 'premium': opt['mid'],
            'expiry': opt['expiration'], 'entry_date': dt,
        }
        self.state = 'CSP_OPEN'

    def _handle_csp_expiry(self, dt, close, regime):
        p = self.open_put
        t = self.cur_trade_put
        t.exit_date = dt
        t.exit_regime = regime
        t.exit_price = close
        t.days_held = _days(p['entry_date'], dt)
        if close > p['strike']:
            # OTM expiry: keep premium
            t.exit_reason = 'csp_otm'
            t.pnl = p['premium'] * 100
            self.trades.append(t)
            self.open_put = None
            self.cur_trade_put = None
            self.state = 'CASH'
        else:
            # ITM: assigned
            self.cash -= p['strike'] * 100
            self.shares = 100
            self.cost_basis = p['strike'] - p['premium']
            t.exit_reason = 'csp_assigned'
            t.pnl = p['premium'] * 100 + (close - p['strike']) * 100  # premium + unrealized
            self.trades.append(t)
            self.open_put = None
            self.cur_trade_put = None
            self.state = 'HOLDING'

    def _try_open_cc(self, dt, close, regime, options):
        if dt not in options['call']: return
        opt = options['call'][dt]
        if not opt['mid'] or opt['mid'] <= 0: return
        if opt['strike'] < self.cost_basis: return  # don't lock in loss

        prem = opt['mid'] * 100
        self.cash += prem
        self.trade_counter += 1
        self.cur_trade_call = Trade(
            strategy=self.name, trade_id=self.trade_counter,
            entry_date=dt, entry_regime=regime, entry_price=opt['mid'],
            leg_type='short_call', strike=opt['strike'], expiry=opt['expiration'],
        )
        self.open_call = {
            'strike': opt['strike'], 'premium': opt['mid'],
            'expiry': opt['expiration'], 'entry_date': dt,
        }
        self.state = 'HOLDING_CC'

    def _handle_cc_expiry(self, dt, close, regime):
        c = self.open_call
        t = self.cur_trade_call
        t.exit_date = dt
        t.exit_regime = regime
        t.exit_price = close
        t.days_held = _days(c['entry_date'], dt)
        if close < c['strike']:
            # OTM: keep premium, hold stock
            t.exit_reason = 'cc_otm'
            t.pnl = c['premium'] * 100
            self.trades.append(t)
            self.open_call = None
            self.cur_trade_call = None
            self.state = 'HOLDING'
        else:
            # Called away
            proceeds = c['strike'] * 100
            self.cash += proceeds
            stock_pnl = (c['strike'] - self.cost_basis) * 100
            t.exit_reason = 'cc_called_away'
            t.pnl = c['premium'] * 100 + stock_pnl
            self.trades.append(t)
            self.shares = 0
            self.cost_basis = 0.0
            self.open_call = None
            self.cur_trade_call = None
            self.state = 'CASH'

    def _wheel_stop(self, dt, close, regime):
        proceeds = self.shares * close
        self.cash += proceeds
        stock_pnl = (close - self.cost_basis) * self.shares
        # Close any open call as expired worthless (premium already booked)
        if self.cur_trade_call:
            t = self.cur_trade_call
            t.exit_date = dt
            t.exit_regime = regime
            t.exit_price = close
            t.days_held = _days(self.open_call['entry_date'], dt)
            t.exit_reason = 'wheel_stop'
            t.pnl = self.open_call['premium'] * 100
            self.trades.append(t)
            self.cur_trade_call = None
            self.open_call = None
        # Record the stock leg as its own trade for clarity
        self.trade_counter += 1
        t = Trade(
            strategy=self.name, trade_id=self.trade_counter,
            entry_date=dt, entry_regime=regime, entry_price=self.cost_basis,
            leg_type='long_stock', exit_date=dt, exit_regime=regime,
            exit_price=close, exit_reason='wheel_stop', pnl=stock_pnl, days_held=0,
        )
        self.trades.append(t)
        self.shares = 0
        self.cost_basis = 0.0
        self.state = 'CASH'


# -- Strategy 2: PURE_CSP ---------------------------------------------------

class PureCSPStrategy(Strategy):
    """Sell 35-45 DTE 8% OTM puts. Close at 50% profit. If ITM near expiry,
    accept assignment then immediately sell stock at market the next day."""
    name = 'PURE_CSP'

    def __init__(self, starting_cash):
        super().__init__(starting_cash)
        self.state = 'CASH'   # CASH | OPEN | LIQUIDATING
        self.open_put = None
        self.cur_trade = None
        self.holding_shares = 0
        self.holding_basis = 0.0

    def portfolio_value(self, close_price):
        return self.cash + self.holding_shares * close_price

    def step(self, dt, close, regime, options):
        # Liquidate any held shares ASAP at next available close
        if self.holding_shares > 0:
            self.cash += self.holding_shares * close
            # Trade was already recorded at assignment; this is just cleanup
            self.holding_shares = 0
            self.holding_basis = 0.0
            self.state = 'CASH'

        # Handle expiry or 50% profit close
        if self.state == 'OPEN':
            self._manage_open(dt, close, regime, options)

        # Open new
        if self.state == 'CASH':
            self._try_open(dt, close, regime, options)

    def _try_open(self, dt, close, regime, options):
        if dt not in options['put']: return
        opt = options['put'][dt]
        if not opt['mid'] or opt['mid'] <= 0: return
        if self.cash < opt['strike'] * 100: return

        prem = opt['mid'] * 100
        self.cash += prem
        self.trade_counter += 1
        self.cur_trade = Trade(
            strategy=self.name, trade_id=self.trade_counter,
            entry_date=dt, entry_regime=regime, entry_price=opt['mid'],
            leg_type='short_put', strike=opt['strike'], expiry=opt['expiration'],
        )
        self.open_put = {
            'strike': opt['strike'], 'entry_premium': opt['mid'],
            'expiry': opt['expiration'], 'entry_date': dt,
        }
        self.state = 'OPEN'

    def _manage_open(self, dt, close, regime, options):
        p = self.open_put
        # Lookup current put price (use today's put record as proxy for current value)
        cur_premium = None
        if dt in options['put']:
            cur_premium = options['put'][dt].get('mid')

        # 50% profit close
        if cur_premium is not None and cur_premium <= p['entry_premium'] * (1 - CSP_PROFIT_TAKE):
            # Buy back at current premium
            self.cash -= cur_premium * 100
            t = self.cur_trade
            t.exit_date = dt
            t.exit_regime = regime
            t.exit_price = cur_premium
            t.exit_reason = 'profit_50pct'
            t.pnl = (p['entry_premium'] - cur_premium) * 100
            t.days_held = _days(p['entry_date'], dt)
            self.trades.append(t)
            self.cur_trade = None
            self.open_put = None
            self.state = 'CASH'
            return

        # Expiry
        if dt >= p['expiry']:
            t = self.cur_trade
            t.exit_date = dt
            t.exit_regime = regime
            t.exit_price = close
            t.days_held = _days(p['entry_date'], dt)
            if close > p['strike']:
                t.exit_reason = 'otm_expiry'
                t.pnl = p['entry_premium'] * 100
                self.trades.append(t)
                self.cur_trade = None
                self.open_put = None
                self.state = 'CASH'
            else:
                # Assigned, then liquidate next day
                self.cash -= p['strike'] * 100
                self.holding_shares = 100
                self.holding_basis = p['strike'] - p['entry_premium']
                t.exit_reason = 'assigned_liquidate'
                # PnL = premium - (strike - close)*100 [intrinsic loss at expiry]
                t.pnl = p['entry_premium'] * 100 - (p['strike'] - close) * 100
                self.trades.append(t)
                self.cur_trade = None
                self.open_put = None
                # holding_shares is liquidated next step


# -- Strategy 3: LONG_CALL --------------------------------------------------

class LongCallStrategy(Strategy):
    """Buy 35-45 DTE 8% OTM call. Close at 100% gain, 50% loss, or expiry."""
    name = 'LONG_CALL'

    def __init__(self, starting_cash):
        super().__init__(starting_cash)
        self.state = 'CASH'   # CASH | OPEN
        self.open_call = None
        self.cur_trade = None
        self.position_size_pct = 0.10   # use 10% of capital per call trade

    def portfolio_value(self, close_price):
        # We don't mark-to-market the open call (we don't have daily option prices)
        # Position value = cash; the trade either pays off at exit or it doesn't.
        return self.cash

    def step(self, dt, close, regime, options):
        if self.state == 'OPEN':
            self._manage_open(dt, close, regime, options)
        if self.state == 'CASH':
            self._try_open(dt, close, regime, options)

    def _try_open(self, dt, close, regime, options):
        if dt not in options['call']: return
        opt = options['call'][dt]
        if not opt['mid'] or opt['mid'] <= 0: return
        # Buy at mid (since we're selling at mid elsewhere; symmetric)
        budget = self.cash * self.position_size_pct
        cost_per_contract = opt['mid'] * 100
        if cost_per_contract <= 0 or cost_per_contract > budget: return
        contracts = max(1, int(budget // cost_per_contract))
        cost = contracts * cost_per_contract
        if cost > self.cash: return

        self.cash -= cost
        self.trade_counter += 1
        self.cur_trade = Trade(
            strategy=self.name, trade_id=self.trade_counter,
            entry_date=dt, entry_regime=regime, entry_price=opt['mid'],
            leg_type='long_call', strike=opt['strike'], expiry=opt['expiration'],
            contracts=contracts,
        )
        self.open_call = {
            'strike': opt['strike'], 'entry_premium': opt['mid'],
            'expiry': opt['expiration'], 'entry_date': dt,
            'contracts': contracts,
        }
        self.state = 'OPEN'

    def _manage_open(self, dt, close, regime, options):
        """Hold to expiry only. We don't have daily option price history for
        the SPECIFIC contract we bought (DB stores one strike per date, which
        was the 8% OTM call AS OF that date -- not the strike we entered last
        month). So profit/stop checks would compare against a different option
        entirely. Honest fix: just hold to expiry and value at intrinsic."""
        c = self.open_call
        if dt >= c['expiry']:
            intrinsic = max(0.0, close - c['strike'])
            self._close(dt, regime, intrinsic, 'expiry')

    def _close(self, dt, regime, exit_premium, reason):
        c = self.open_call
        proceeds = exit_premium * 100 * c['contracts']
        self.cash += proceeds
        t = self.cur_trade
        t.exit_date = dt
        t.exit_regime = regime
        t.exit_price = exit_premium
        t.exit_reason = reason
        t.pnl = (exit_premium - c['entry_premium']) * 100 * c['contracts']
        t.days_held = _days(c['entry_date'], dt)
        self.trades.append(t)
        self.open_call = None
        self.cur_trade = None
        self.state = 'CASH'


# -- Strategy 4: BUY_HOLD ---------------------------------------------------

class BuyHoldStrategy(Strategy):
    name = 'BUY_HOLD'

    def __init__(self, starting_cash):
        super().__init__(starting_cash)
        self.shares = 0
        self.entered = False
        self.entry_price = 0.0

    def portfolio_value(self, close_price):
        return self.cash + self.shares * close_price

    def step(self, dt, close, regime, options):
        if not self.entered and close > 0:
            self.shares = int(self.cash // close)
            self.cash -= self.shares * close
            self.entry_price = close
            self.trade_counter += 1
            t = Trade(
                strategy=self.name, trade_id=1,
                entry_date=dt, entry_regime=regime, entry_price=close,
                leg_type='long_stock', contracts=self.shares,
            )
            self.trades.append(t)
            self.entered = True


# -- Helpers ----------------------------------------------------------------

def _days(d1: str, d2: str) -> int:
    return (datetime.strptime(d2, '%Y-%m-%d').date()
            - datetime.strptime(d1, '%Y-%m-%d').date()).days


def calc_metrics(values: pd.Series, starting: float) -> dict:
    daily_ret = values.pct_change().dropna()
    if len(daily_ret) < 2:
        return {}
    total_return = (values.iloc[-1] / starting) - 1
    years = len(daily_ret) / TRADING_DAYS
    cagr = (values.iloc[-1] / starting) ** (1 / years) - 1 if years > 0 else 0
    excess = daily_ret - (RISK_FREE_RATE / TRADING_DAYS)
    # Use a meaningful std threshold (1e-6) instead of >0 to avoid floating
    # point dust producing astronomical Sharpe values on flat equity curves.
    sharpe = (excess.mean() / excess.std()) * math.sqrt(TRADING_DAYS) if excess.std() > 1e-6 else 0
    cumulative = values / values.expanding().max()
    max_dd = cumulative.min() - 1
    return {
        'total_return_pct': round(total_return * 100, 2),
        'annual_return_pct': round(cagr * 100, 2),
        'sharpe_ratio': round(sharpe, 3),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'final_value': round(float(values.iloc[-1]), 2),
    }


# -- Main -------------------------------------------------------------------

def get_eligible_tickers(db_path: str) -> list:
    """Return tickers that have BOTH puts and calls in the DB."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT symbol FROM options_history
        GROUP BY symbol
        HAVING COUNT(DISTINCT side) = 2
        ORDER BY symbol
    """).fetchall()
    conn.close()
    return [r[0] for r in rows]


def filter_options_to_window(options: dict, start: str, end: str) -> dict:
    """Trim options dict to only dates within [start, end]."""
    out = {'put': {}, 'call': {}}
    for side in ('put', 'call'):
        for dt, rec in options[side].items():
            if start <= dt <= end:
                out[side][dt] = rec
    return out


def run_one_ticker(ticker: str, options: dict, prices: pd.DataFrame,
                   regime_df: pd.DataFrame, starting_capital: float) -> dict:
    """Run all 4 strategies for one ticker. Returns trades, equity, metrics."""
    strategies = [
        WheelStrategy(starting_capital),
        PureCSPStrategy(starting_capital),
        LongCallStrategy(starting_capital),
        BuyHoldStrategy(starting_capital),
    ]

    dates = list(prices.index)
    for dt in dates:
        if dt not in regime_df.index:
            continue
        close = float(prices.loc[dt, 'Close'])
        regime = regime_df.loc[dt, 'regime']
        for s in strategies:
            s.step(dt, close, regime, options)
            s.record_equity(dt, close)

    # Tag every trade with the ticker
    trades = []
    for s in strategies:
        for t in s.trades:
            d = asdict(t)
            d['ticker'] = ticker
            trades.append(d)

    # Per-strategy equity series
    equity = {}
    for s in strategies:
        if s.equity_curve:
            ec = pd.DataFrame(s.equity_curve).set_index('date')['value']
            equity[s.name] = ec

    # Per-strategy metrics
    metrics = {}
    for s in strategies:
        if s.name in equity:
            metrics[s.name] = calc_metrics(equity[s.name], starting_capital)
            metrics[s.name]['n_trades'] = sum(1 for t in trades
                                              if t['strategy'] == s.name)
            metrics[s.name]['starting_capital'] = starting_capital

    return {'trades': trades, 'equity': equity, 'metrics': metrics}


def main():
    import math as _math
    print(f"[*] Multi-Strategy Backtester -- Multi-Ticker")
    print(f"   DB:        {DB_PATH}")
    print(f"   Capital:   per-ticker scaled to afford 1 contract (min ${MIN_STARTING_CAPITAL:,.0f})")
    print(f"   Window:    {ANALYSIS_START} -> {ANALYSIS_END} (1 year, apples-to-apples)")
    print()

    tickers = get_eligible_tickers(DB_PATH)
    print(f"Found {len(tickers)} tickers with both puts and calls:")
    print(f"   {', '.join(tickers)}")
    print()

    if not tickers:
        print("[X] No tickers with both puts and calls. Aborting.")
        sys.exit(1)

    # Load daily score data from backtester (if available)
    print("Loading model scores...")
    scores_lookup = load_scores(SCORES_CSV)
    print()

    # Build regime timeline once (covers the analysis window with EMA warmup)
    regime_start = (datetime.strptime(ANALYSIS_START, '%Y-%m-%d').date()
                    - timedelta(days=90)).strftime('%Y-%m-%d')
    regime_end = (datetime.strptime(ANALYSIS_END, '%Y-%m-%d').date()
                  + timedelta(days=60)).strftime('%Y-%m-%d')
    print(f"Building regime timeline ({regime_start} -> {regime_end})...")
    regime_df = build_regime_timeline(regime_start, regime_end)
    print(f"   {len(regime_df)} regime days")
    print(f"   Regime distribution in analysis window:")
    in_window = regime_df.loc[regime_df.index >= ANALYSIS_START]
    in_window = in_window.loc[in_window.index <= ANALYSIS_END]
    for r, c in in_window['regime'].value_counts().items():
        print(f"     {r:<25} {c}")
    print()

    # Run each ticker
    all_trades = []
    all_equity = {}        # {(ticker, strategy): series}
    all_metrics = {}       # {ticker: {strategy: metrics}}
    summary_rows = []      # for headline table

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...", end=' ', flush=True)
        try:
            options_full = load_options(DB_PATH, ticker)
            options = filter_options_to_window(options_full, ANALYSIS_START, ANALYSIS_END)

            if not options['put'] or not options['call']:
                print(f"SKIP (no data in window: puts={len(options['put'])}, calls={len(options['call'])})")
                continue

            # Price data, window only.
            # yfinance treats `end` as EXCLUSIVE, so pad by 1 day to include ANALYSIS_END.
            yf_end = (datetime.strptime(ANALYSIS_END, '%Y-%m-%d').date()
                      + timedelta(days=1)).strftime('%Y-%m-%d')
            prices = load_prices(ticker, ANALYSIS_START, yf_end)
            if prices.empty:
                print(f"SKIP (no price data)")
                continue

            # Compute starting capital: enough to afford 1 contract at first put strike + 10% buffer
            first_put_date = sorted(options['put'].keys())[0]
            first_put_strike = options['put'][first_put_date]['strike']
            required = first_put_strike * 100 * 1.10
            # Round up to nearest $5,000 for clean numbers
            ticker_capital = max(MIN_STARTING_CAPITAL,
                                 _math.ceil(required / 5000.0) * 5000.0)

            result = run_one_ticker(ticker, options, prices, regime_df, ticker_capital)
            all_trades.extend(result['trades'])
            for sname, ec in result['equity'].items():
                all_equity[(ticker, sname)] = ec
            all_metrics[ticker] = result['metrics']
            n = len(result['trades'])
            print(f"OK (capital=${ticker_capital:,.0f}, {n} trades, "
                  f"puts={len(options['put'])}, calls={len(options['call'])})")

            # Build summary row per (ticker, strategy)
            for sname, m in result['metrics'].items():
                if not m: continue
                summary_rows.append({
                    'ticker': ticker, 'strategy': sname,
                    **{k: v for k, v in m.items()}
                })
        except Exception as e:
            print(f"FAIL: {e}")
            continue

    if not all_trades:
        print("\n[X] No trades produced. Check data and date window.")
        sys.exit(1)

    # -- Outputs --
    print(f"\nWriting outputs...")

    trades_df = pd.DataFrame(all_trades)

    # Tag every trade with its entry-day score (from backtest_daily_scores.csv)
    if scores_lookup:
        trades_df['entry_score'] = trades_df.apply(
            lambda r: scores_lookup.get((r['ticker'], r['entry_date'])), axis=1)
        trades_df['entry_score_bucket'] = trades_df['entry_score'].apply(bucket_score)
        n_tagged = trades_df['entry_score'].notna().sum()
        print(f"   Tagged {n_tagged:,}/{len(trades_df):,} trades with scores "
              f"({100*n_tagged/len(trades_df):.1f}%)")
    else:
        trades_df['entry_score'] = None
        trades_df['entry_score_bucket'] = 'NO_SCORE'

    trades_df.to_csv('all_trades.csv', index=False)
    print(f"   all_trades.csv ({len(trades_df)} rows)")

    headline_df = pd.DataFrame(summary_rows)
    headline_df.to_csv('headline_per_ticker_strategy.csv', index=False)
    print(f"   headline_per_ticker_strategy.csv ({len(headline_df)} rows)")

    regime_df[['vix', 'spy', 'regime']].to_csv('regime_timeline.csv')
    print(f"   regime_timeline.csv")

    # Strategy x Regime matrix (AGGREGATED across all tickers)
    matrix_rows = []
    for strat in trades_df['strategy'].unique():
        s_trades = trades_df[trades_df['strategy'] == strat]
        for regime in sorted(s_trades['entry_regime'].dropna().unique()):
            sub = s_trades[s_trades['entry_regime'] == regime]
            wins = sub[sub['pnl'] > 0]
            matrix_rows.append({
                'strategy': strat,
                'regime': regime,
                'n_trades': len(sub),
                'n_tickers': sub['ticker'].nunique(),
                'win_rate_pct': round(100 * len(wins) / len(sub), 1) if len(sub) else 0,
                'avg_pnl': round(sub['pnl'].mean(), 2),
                'total_pnl': round(sub['pnl'].sum(), 2),
                'avg_days_held': round(sub['days_held'].mean(), 1),
            })
    matrix_df = pd.DataFrame(matrix_rows)
    matrix_df.to_csv('strategy_regime_matrix.csv', index=False)
    print(f"   strategy_regime_matrix.csv ({len(matrix_df)} rows)")

    # Ticker x Strategy x Regime matrix (FULL DETAIL)
    full_rows = []
    for tkr in sorted(trades_df['ticker'].unique()):
        t_trades = trades_df[trades_df['ticker'] == tkr]
        for strat in t_trades['strategy'].unique():
            s_trades = t_trades[t_trades['strategy'] == strat]
            for regime in sorted(s_trades['entry_regime'].dropna().unique()):
                sub = s_trades[s_trades['entry_regime'] == regime]
                wins = sub[sub['pnl'] > 0]
                full_rows.append({
                    'ticker': tkr, 'strategy': strat, 'regime': regime,
                    'n_trades': len(sub),
                    'win_rate_pct': round(100 * len(wins) / len(sub), 1) if len(sub) else 0,
                    'avg_pnl': round(sub['pnl'].mean(), 2),
                    'total_pnl': round(sub['pnl'].sum(), 2),
                })
    full_df = pd.DataFrame(full_rows)
    full_df.to_csv('ticker_strategy_regime_matrix.csv', index=False)
    print(f"   ticker_strategy_regime_matrix.csv ({len(full_df)} rows)")

    # Strategy x Regime x Score_Bucket matrix (the score-aware view)
    score_matrix_rows = []
    for strat in sorted(trades_df['strategy'].unique()):
        s_trades = trades_df[trades_df['strategy'] == strat]
        for regime in sorted(s_trades['entry_regime'].dropna().unique()):
            r_trades = s_trades[s_trades['entry_regime'] == regime]
            for bucket in ['LOW', 'MEDIUM', 'HIGH', 'NO_SCORE']:
                sub = r_trades[r_trades['entry_score_bucket'] == bucket]
                if sub.empty:
                    continue
                wins = sub[sub['pnl'] > 0]
                score_matrix_rows.append({
                    'strategy':       strat,
                    'regime':         regime,
                    'score_bucket':   bucket,
                    'n_trades':       len(sub),
                    'n_tickers':      sub['ticker'].nunique(),
                    'win_rate_pct':   round(100 * len(wins) / len(sub), 1) if len(sub) else 0,
                    'avg_pnl':        round(sub['pnl'].mean(), 2),
                    'total_pnl':      round(sub['pnl'].sum(), 2),
                    'avg_score':      round(sub['entry_score'].mean(), 3) if sub['entry_score'].notna().any() else None,
                    'avg_days_held':  round(sub['days_held'].mean(), 1),
                })
    score_matrix_df = pd.DataFrame(score_matrix_rows)
    score_matrix_df.to_csv('strategy_regime_score_matrix.csv', index=False)
    print(f"   strategy_regime_score_matrix.csv ({len(score_matrix_df)} rows)")

    summary = {
        'analysis_window': {'start': ANALYSIS_START, 'end': ANALYSIS_END},
        'min_starting_capital': MIN_STARTING_CAPITAL,
        'n_tickers': len(all_metrics),
        'tickers': list(all_metrics.keys()),
        'per_ticker_metrics': all_metrics,
    }
    with open('multi_strategy_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"   multi_strategy_summary.json")

    # -- Console summary --
    print(f"\n{'='*92}")
    print(f"[+] Headline metrics (1 year, per-ticker capital scaled to afford 1 contract)")
    print(f"{'='*92}")
    print(f"{'Ticker':<8}{'Strategy':<12}{'Capital':>11}{'Final $':>13}"
          f"{'Ret %':>9}{'Sharpe':>9}{'MaxDD %':>10}{'Trades':>9}")
    print(f"{'-'*92}")
    for _, row in headline_df.iterrows():
        cap = row.get('starting_capital', MIN_STARTING_CAPITAL)
        print(f"{row['ticker']:<8}{row['strategy']:<12}"
              f"${cap:>10,.0f}"
              f" ${row['final_value']:>11,.0f}"
              f"{row['total_return_pct']:>8.1f}%"
              f"{row['sharpe_ratio']:>9.3f}"
              f"{row['max_drawdown_pct']:>9.1f}%"
              f"{row['n_trades']:>9}")

    print(f"\n{'='*84}")
    print(f"[>] Aggregate Strategy x Regime Matrix (all tickers combined)")
    print(f"{'='*84}")
    print(f"{'Strategy':<12}{'Regime':<22}{'N':>6}{'Tkrs':>6}{'Win%':>8}{'AvgPnL':>10}{'Total':>12}")
    print(f"{'-'*84}")
    for _, row in matrix_df.iterrows():
        print(f"{row['strategy']:<12}{row['regime']:<22}{row['n_trades']:>6}{row['n_tickers']:>6}"
              f"{row['win_rate_pct']:>7.1f}%"
              f"${row['avg_pnl']:>9,.0f}"
              f"${row['total_pnl']:>11,.0f}")

    print(f"\n{'='*84}")
    print(f"[#] Best strategy per regime (by total P&L, aggregated)")
    print(f"{'='*84}")
    for regime in sorted(matrix_df['regime'].unique()):
        sub = matrix_df[matrix_df['regime'] == regime]
        if sub.empty: continue
        best = sub.loc[sub['total_pnl'].idxmax()]
        print(f"   {regime:<25} -> {best['strategy']:<12} "
              f"(${best['total_pnl']:>10,.0f} over {best['n_trades']} trades "
              f"on {best['n_tickers']} tickers, {best['win_rate_pct']:.0f}% win rate)")

    print(f"\n{'='*84}")
    print(f"[#] Best strategy per ticker (overall)")
    print(f"{'='*84}")
    for tkr in sorted(headline_df['ticker'].unique()):
        sub = headline_df[headline_df['ticker'] == tkr]
        if sub.empty: continue
        best = sub.loc[sub['total_return_pct'].idxmax()]
        print(f"   {tkr:<8} -> {best['strategy']:<12} "
              f"({best['total_return_pct']:>+6.1f}% return, "
              f"Sharpe {best['sharpe_ratio']:.2f})")

    # Score-bucket section (only meaningful cells with N>=10 trades)
    if not score_matrix_df.empty and scores_lookup:
        print(f"\n{'='*92}")
        print(f"[>] Strategy x Regime x Score Bucket  (cells with N>=10 trades)")
        print(f"{'='*92}")
        print(f"{'Strategy':<12}{'Regime':<22}{'Bucket':<11}{'N':>6}{'Tkrs':>6}"
              f"{'Win%':>7}{'AvgScore':>10}{'AvgPnL':>10}{'Total':>12}")
        print(f"{'-'*92}")
        meaningful = score_matrix_df[score_matrix_df['n_trades'] >= 10]
        for _, row in meaningful.sort_values(['strategy', 'regime', 'score_bucket']).iterrows():
            score_str = f"{row['avg_score']:.3f}" if row['avg_score'] is not None and row['avg_score'] == row['avg_score'] else 'n/a'
            print(f"{row['strategy']:<12}{row['regime']:<22}{row['score_bucket']:<11}"
                  f"{row['n_trades']:>6}{row['n_tickers']:>6}"
                  f"{row['win_rate_pct']:>6.1f}%"
                  f"{score_str:>10}"
                  f"${row['avg_pnl']:>9,.0f}"
                  f"${row['total_pnl']:>11,.0f}")
        if meaningful.empty:
            print(f"   (no cells with N>=10 -- full data in strategy_regime_score_matrix.csv)")

        print(f"\n{'='*92}")
        print(f"[#] Best score bucket per (strategy, regime) -- where conviction pays off")
        print(f"{'='*92}")
        for strat in sorted(score_matrix_df['strategy'].unique()):
            for regime in sorted(score_matrix_df[score_matrix_df['strategy']==strat]['regime'].unique()):
                cells = score_matrix_df[
                    (score_matrix_df['strategy']==strat) &
                    (score_matrix_df['regime']==regime) &
                    (score_matrix_df['n_trades']>=5)
                ]
                if cells.empty: continue
                # Compare avg_pnl across buckets within this (strategy, regime)
                buckets_str = ' | '.join(
                    f"{r['score_bucket']}=${r['avg_pnl']:+.0f} (n={r['n_trades']})"
                    for _, r in cells.iterrows()
                )
                best = cells.loc[cells['avg_pnl'].idxmax()]
                print(f"   {strat:<12} {regime:<22} -> best: {best['score_bucket']:<8} "
                      f"avg ${best['avg_pnl']:+,.0f}")
                print(f"   {'':<35}    {buckets_str}")


if __name__ == '__main__':
    main()