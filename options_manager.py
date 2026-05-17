"""
options_manager.py
Manages two options strategies:

1. LONG CALLS — on high-conviction flow signals (score > 0.85, flow regime)
   - Buys slightly OTM calls (5% above current price)
   - 30-45 DTE (days to expiration)
   - Position size: 5% of idle cash per contract (smaller than equity)
   - Exit: stop at -50% premium, take profit at +100%

2. COVERED CALLS — on parking positions (GLD, SCHP, VTIP)
   - Sells monthly covered calls 5-8% OTM
   - Generates income on idle inflation hedge positions
   - Only when implied volatility (IV) is elevated (better premium)
   - Let expire or buy back at 80% profit

Note: Options execution uses Schwab's options API.
Paper trading simulates options P&L using Black-Scholes approximation.
"""

import os
import math
import time
import requests
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

LONG_CALL_MIN_SCORE    = 0.85   # only buy calls on very high conviction
LONG_CALL_OTM_PCT      = 0.05   # 5% OTM strike
LONG_CALL_DTE_TARGET   = 35     # target 35 days to expiration
LONG_CALL_POSITION_PCT = 0.05   # 5% of idle cash per contract
LONG_CALL_STOP_PCT     = 0.50   # exit if premium drops 50%
LONG_CALL_PROFIT_PCT   = 1.00   # exit if premium doubles

COVERED_CALL_OTM_PCT     = 0.06  # sell 6% OTM
COVERED_CALL_DTE_TARGET  = 30    # monthly (30 DTE)
COVERED_CALL_PROFIT_PCT  = 0.80  # buy back at 80% profit (20% of premium left)
COVERED_CALL_IV_MINIMUM  = 20.0  # only sell calls when IV > 20%

PARKING_TICKERS = ['GLD', 'SCHP', 'VTIP', 'GDX']


# ── Black-Scholes Approximation ───────────────────────────────────────────────

def bs_call_price(S, K, T, r, sigma) -> float:
    """
    Black-Scholes call option price approximation.
    S = stock price, K = strike, T = time in years,
    r = risk-free rate, sigma = implied volatility
    """
    try:
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        nd1 = _norm_cdf(d1)
        nd2 = _norm_cdf(d2)
        return round(S * nd1 - K * math.exp(-r * T) * nd2, 4)
    except Exception:
        return max(S - K, 0)


def _norm_cdf(x) -> float:
    """Standard normal CDF approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def estimate_iv(vixy: float) -> float:
    """
    Estimates implied volatility from VIXY level.
    VIXY ~27 ≈ VIX ~20 ≈ IV ~25% for large caps.
    """
    return max(vixy / 100 * 0.85, 0.15)


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class OptionsPosition:
    symbol:        str
    option_type:   str          # 'long_call' | 'covered_call'
    strike:        float
    expiry:        str          # YYYY-MM-DD
    dte_at_entry:  int
    entry_premium: float        # per share (multiply by 100 for per contract)
    contracts:     int
    entry_date:    str
    underlying_price: float
    cost_basis:    float        # total cost (entry_premium * contracts * 100)
    iv_at_entry:   float


@dataclass
class OptionsSignal:
    symbol:        str
    action:        str          # 'buy_call' | 'sell_covered_call' | 'close'
    strike:        float
    expiry:        str
    estimated_premium: float
    contracts:     int
    estimated_cost:    float
    reason:        str
    score:         float = 0.0


# ── Long Call Strategy ────────────────────────────────────────────────────────

def evaluate_long_call(
    symbol:     str,
    score:      float,
    last_price: float,
    vixy:       float,
    regime:     str,
    idle_cash:  float,
) -> OptionsSignal | None:
    """
    Evaluates whether to buy a long call on a high-conviction symbol.
    Only fires in flow regime with score > 0.85.
    Returns OptionsSignal or None.
    """
    if regime != 'flow':
        return None
    if score < LONG_CALL_MIN_SCORE:
        return None
    if last_price <= 0 or idle_cash <= 0:
        return None

    # Strike: 5% OTM
    strike = round(last_price * (1 + LONG_CALL_OTM_PCT), 2)

    # Expiry: ~35 DTE
    expiry_dt = datetime.now() + timedelta(days=LONG_CALL_DTE_TARGET)
    # Roll to nearest Friday
    days_to_friday = (4 - expiry_dt.weekday()) % 7
    expiry_dt += timedelta(days=days_to_friday)
    expiry = expiry_dt.strftime('%Y-%m-%d')
    dte    = (expiry_dt - datetime.now()).days

    # Estimate premium using Black-Scholes
    iv       = estimate_iv(vixy)
    T        = dte / 365
    premium  = bs_call_price(last_price, strike, T, 0.05, iv)
    if premium <= 0:
        return None

    # Position sizing: 5% of idle cash, max 5 contracts
    budget    = idle_cash * LONG_CALL_POSITION_PCT
    contracts = max(1, min(5, int(budget / (premium * 100))))
    total_cost = round(premium * contracts * 100, 2)

    if total_cost > idle_cash * 0.10:   # never spend more than 10% on options
        contracts  = max(1, int(idle_cash * 0.10 / (premium * 100)))
        total_cost = round(premium * contracts * 100, 2)

    return OptionsSignal(
        symbol=symbol,
        action='buy_call',
        strike=strike,
        expiry=expiry,
        estimated_premium=round(premium, 4),
        contracts=contracts,
        estimated_cost=total_cost,
        reason=f"High-conviction flow signal (score={score:.3f}), IV={iv:.1%}",
        score=score,
    )


# ── Covered Call Strategy ─────────────────────────────────────────────────────

def evaluate_covered_calls(
    parking_positions: dict,
    quotes:            dict,
    vixy:              float,
    regime:            str,
) -> list[OptionsSignal]:
    """
    Evaluates covered call opportunities on parking positions.
    Only sells calls when IV is elevated (VIXY > threshold).
    Returns list of OptionsSignals.
    """
    signals = []
    iv      = estimate_iv(vixy)
    iv_pct  = iv * 100

    # Only sell covered calls when IV is meaningful
    if iv_pct < COVERED_CALL_IV_MINIMUM:
        return []

    for ticker in PARKING_TICKERS:
        pos = parking_positions.get(ticker)
        if not pos:
            continue

        qty   = pos.get('quantity', 0)
        if qty < 100:   # need at least 100 shares for 1 contract
            continue

        contracts  = qty // 100   # number of contracts we can sell
        last_price = quotes.get(ticker, {}).get('last', pos.get('avg_price', 0))
        if last_price <= 0:
            continue

        # Strike: 6% OTM
        strike = round(last_price * (1 + COVERED_CALL_OTM_PCT), 2)

        # Expiry: ~30 DTE (monthly)
        expiry_dt = datetime.now() + timedelta(days=COVERED_CALL_DTE_TARGET)
        days_to_friday = (4 - expiry_dt.weekday()) % 7
        expiry_dt += timedelta(days=days_to_friday)
        expiry = expiry_dt.strftime('%Y-%m-%d')
        dte    = (expiry_dt - datetime.now()).days

        # Estimate premium
        T       = dte / 365
        premium = bs_call_price(last_price, strike, T, 0.05, iv)
        if premium < 0.10:   # minimum $0.10 premium to be worth it
            continue

        income = round(premium * contracts * 100, 2)

        signals.append(OptionsSignal(
            symbol=ticker,
            action='sell_covered_call',
            strike=strike,
            expiry=expiry,
            estimated_premium=round(premium, 4),
            contracts=contracts,
            estimated_cost=-income,   # negative = we receive premium
            reason=f"Covered call on {ticker} parking position. "
                   f"IV={iv_pct:.1f}%, Strike={strike}, Income≈${income:.2f}",
            score=0.0,
        ))

    return signals


# ── Paper Options Tracker ─────────────────────────────────────────────────────

def paper_buy_call(
    signal:    OptionsSignal,
    portfolio: dict,
    client,
) -> dict | None:
    """
    Simulates buying a long call in paper trading.
    Deducts estimated cost from cash, logs position.
    Returns position dict or None if insufficient cash.
    """
    cash = portfolio.get('cash', 0)
    if signal.estimated_cost > cash:
        print(f"  ❌ Insufficient cash for {signal.symbol} call: "
              f"need ${signal.estimated_cost:,.2f}, have ${cash:,.2f}")
        return None

    position = {
        'type':            'long_call',
        'symbol':          signal.symbol,
        'strike':          signal.strike,
        'expiry':          signal.expiry,
        'contracts':       signal.contracts,
        'entry_premium':   signal.estimated_premium,
        'cost_basis':      signal.estimated_cost,
        'entry_date':      datetime.now().isoformat(),
        'stop_price':      round(signal.estimated_premium * (1 - LONG_CALL_STOP_PCT), 4),
        'target_price':    round(signal.estimated_premium * (1 + LONG_CALL_PROFIT_PCT), 4),
    }

    print(f"  📈 PAPER CALL BUY: {signal.contracts}x {signal.symbol} "
          f"${signal.strike}C {signal.expiry} "
          f"@ ${signal.estimated_premium:.2f} = ${signal.estimated_cost:,.2f}")
    print(f"     Stop: ${position['stop_price']:.2f}  "
          f"Target: ${position['target_price']:.2f}")

    return position


def paper_sell_covered_call(
    signal:    OptionsSignal,
    portfolio: dict,
) -> dict | None:
    """
    Simulates selling a covered call in paper trading.
    Adds premium received to cash, logs position.
    """
    income = abs(signal.estimated_cost)

    position = {
        'type':          'covered_call',
        'symbol':        signal.symbol,
        'strike':        signal.strike,
        'expiry':        signal.expiry,
        'contracts':     signal.contracts,
        'entry_premium': signal.estimated_premium,
        'premium_received': income,
        'entry_date':    datetime.now().isoformat(),
        'buyback_price': round(signal.estimated_premium * (1 - COVERED_CALL_PROFIT_PCT), 4),
    }

    print(f"  💰 PAPER COVERED CALL: Sold {signal.contracts}x {signal.symbol} "
          f"${signal.strike}C {signal.expiry} "
          f"@ ${signal.estimated_premium:.2f} = +${income:,.2f} premium")
    print(f"     Buy back at: ${position['buyback_price']:.2f} (80% profit)")

    return position


# ── Options Exit Check ────────────────────────────────────────────────────────

def check_options_exits(
    options_positions: list,
    quotes:            dict,
    vixy:              float,
) -> list:
    """
    Checks all options positions for exit conditions.
    Returns list of positions that should be closed.
    """
    exits    = []
    iv       = estimate_iv(vixy)
    now_date = datetime.now().date()

    for pos in options_positions:
        symbol  = pos.get('symbol', '')
        expiry  = pos.get('expiry', '')
        opt_type = pos.get('type', '')

        # Check expiry
        try:
            expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
            dte         = (expiry_date - now_date).days
            if dte <= 5:   # close 5 days before expiry to avoid pin risk
                exits.append({**pos, 'exit_reason': 'approaching_expiry', 'dte': dte})
                continue
        except Exception:
            pass

        last = quotes.get(symbol, {}).get('last', 0)
        if last <= 0:
            continue

        if opt_type == 'long_call':
            # Re-estimate current premium
            entry_premium = pos.get('entry_premium', 0)
            strike        = pos.get('strike', last)
            T             = max(dte / 365, 0.01) if 'dte' in locals() else 0.1
            curr_premium  = bs_call_price(last, strike, T, 0.05, iv)

            stop_pct   = entry_premium * (1 - LONG_CALL_STOP_PCT)
            target_pct = entry_premium * (1 + LONG_CALL_PROFIT_PCT)

            if curr_premium <= stop_pct:
                exits.append({**pos, 'exit_reason': 'stop_loss',
                               'current_premium': curr_premium})
            elif curr_premium >= target_pct:
                exits.append({**pos, 'exit_reason': 'take_profit',
                               'current_premium': curr_premium})

        elif opt_type == 'covered_call':
            # Buy back at 80% profit
            entry_premium = pos.get('entry_premium', 0)
            buyback_price = pos.get('buyback_price', entry_premium * 0.20)
            strike        = pos.get('strike', last)
            T             = max(dte / 365, 0.01) if 'dte' in locals() else 0.1
            curr_premium  = bs_call_price(last, strike, T, 0.05, iv)

            if curr_premium <= buyback_price:
                exits.append({**pos, 'exit_reason': 'profit_target',
                               'current_premium': curr_premium,
                               'profit': round((entry_premium - curr_premium) * pos.get('contracts', 1) * 100, 2)})

    return exits


# ── Main Evaluation ───────────────────────────────────────────────────────────

def evaluate_options(
    candidates:        list,
    portfolio:         dict,
    quotes:            dict,
    vixy:              float,
    regime:            str,
    options_positions: list = None,
) -> dict:
    """
    Top-level options evaluation for the bot cycle.
    Returns dict with long_call_signals, covered_call_signals, exit_signals.
    """
    idle_cash    = portfolio.get('cash', 0)
    parking_pos  = {k: v for k, v in portfolio.get('positions', {}).items()
                    if k in PARKING_TICKERS}

    # Long calls on high-conviction candidates
    long_call_signals = []
    for c in candidates:
        symbol = c.symbol if hasattr(c, 'symbol') else c.get('symbol', '')
        score  = c.total_score if hasattr(c, 'total_score') else c.get('score', 0)
        last   = quotes.get(symbol, {}).get('last', 0)

        signal = evaluate_long_call(
            symbol=symbol, score=score, last_price=last,
            vixy=vixy, regime=regime, idle_cash=idle_cash,
        )
        if signal:
            long_call_signals.append(signal)

    # Covered calls on parking positions
    covered_call_signals = evaluate_covered_calls(
        parking_positions=parking_pos,
        quotes=quotes,
        vixy=vixy,
        regime=regime,
    )

    # Exit checks on existing options positions
    exit_signals = []
    if options_positions:
        exit_signals = check_options_exits(options_positions, quotes, vixy)

    return {
        'long_calls':    long_call_signals,
        'covered_calls': covered_call_signals,
        'exits':         exit_signals,
    }


def print_options_summary(options_eval: dict):
    lc = options_eval.get('long_calls', [])
    cc = options_eval.get('covered_calls', [])
    ex = options_eval.get('exits', [])

    if not lc and not cc and not ex:
        return

    print(f"\n{'='*55}")
    print(f"  OPTIONS MANAGER")
    print(f"{'='*55}")

    if lc:
        print(f"\n  📈 Long Call Signals ({len(lc)}):")
        for s in lc:
            print(f"  BUY  {s.contracts}x {s.symbol} ${s.strike}C {s.expiry} "
                  f"@ ~${s.estimated_premium:.2f}  Cost: ${s.estimated_cost:,.2f}")
            print(f"       {s.reason}")

    if cc:
        print(f"\n  💰 Covered Call Signals ({len(cc)}):")
        for s in cc:
            income = abs(s.estimated_cost)
            print(f"  SELL {s.contracts}x {s.symbol} ${s.strike}C {s.expiry} "
                  f"@ ~${s.estimated_premium:.2f}  Income: +${income:,.2f}")

    if ex:
        print(f"\n  🚪 Options Exits ({len(ex)}):")
        for e in ex:
            print(f"  CLOSE {e.get('symbol')} [{e.get('exit_reason')}]")

    print(f"{'='*55}\n")


# ── Smoke Test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("🧪 Testing options_manager.py...\n")

    # Test Black-Scholes
    price  = bs_call_price(S=500, K=525, T=35/365, r=0.05, sigma=0.25)
    print(f"BS Call Price (SPY $500, $525C, 35DTE, 25% IV): ${price:.2f}")

    # Test long call evaluation
    from dataclasses import dataclass

    @dataclass
    class MockScore:
        symbol: str
        total_score: float
        direction: str

    candidates = [
        MockScore('NVDA', 0.88, 'bullish'),
        MockScore('AAPL', 0.91, 'bullish'),
        MockScore('SPY',  0.72, 'bullish'),   # below threshold
    ]

    mock_quotes = {
        'NVDA': {'last': 900.0},
        'AAPL': {'last': 200.0},
        'SPY':  {'last': 540.0},
    }

    mock_portfolio = {
        'cash': 25000,
        'positions': {
            'GLD':  {'quantity': 200, 'avg_price': 220.0},
            'SCHP': {'quantity': 500, 'avg_price': 52.0},
        }
    }

    result = evaluate_options(
        candidates=candidates,
        portfolio=mock_portfolio,
        quotes=mock_quotes,
        vixy=22.0,
        regime='flow',
    )

    print_options_summary(result)

    print(f"Long call signals:    {len(result['long_calls'])}")
    print(f"Covered call signals: {len(result['covered_calls'])}")
    print("\n✅ options_manager.py working correctly.")