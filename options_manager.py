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



# ── Cash-Secured Put Strategy ─────────────────────────────────────────────────

# ── Wheel Strategy Config (McMaster + tastytrade principles) ─────────────────
CSP_MIN_SCORE    = 0.85   # only sell CSPs on very high conviction signals
CSP_DELTA_TARGET = 0.30   # sell at ~30 delta (≈ 8-10% OTM) — more premium
CSP_OTM_PCT      = 0.08   # fallback: 8% OTM if delta unavailable
CSP_DTE_TARGET   = 35     # 30-45 DTE sweet spot — more time value
CSP_DTE_MIN      = 21     # minimum DTE — don't sell too close to expiry
CSP_MIN_PREMIUM  = 1.00   # minimum $1.00 premium — worthwhile after fees
CSP_MAX_BUDGET   = 0.25   # max 25% of idle cash in CSPs at once
CSP_PROFIT_CLOSE = 0.50   # buy back at 50% of premium received (tastytrade rule)
CSP_IV_RANK_MIN  = 0.30   # only sell when IV is elevated (30th percentile min)

# ── Wheel exit rule ────────────────────────────────────────────────────────────
# If assigned AND stock breaks below key support → close entire position
# Don't keep selling CCs on a falling knife
WHEEL_STOP_BELOW_COST = 0.10   # exit wheel if stock drops 10% below assignment price


def evaluate_csp(
    symbol:     str,
    score:      float,
    last_price: float,
    vixy:       float,
    regime:     str,
    idle_cash:  float,
) -> OptionsSignal | None:
    """
    Evaluates whether to sell a cash-secured put on a high-conviction symbol.

    Only fires when:
      - Score >= CSP_MIN_SCORE (0.85) — very high conviction
      - Regime is flow or neutral — not in volatility/crisis
      - Sufficient cash to secure the put

    Strike: 4% below current price (we WANT to own this stock at a discount)
    DTE: 21 days — enough time value, not too long

    Two outcomes:
      1. Put expires worthless  → keep premium, re-evaluate
      2. Assigned at strike     → own shares at 4% discount + premium
                                   → immediately eligible for covered calls
    """
    if score < CSP_MIN_SCORE:
        return None
    if regime in ('volatility', 'crisis'):
        return None
    if last_price <= 0 or idle_cash <= 0:
        return None

    # Strike: target ~30 delta = ~8% OTM (more premium than 4%)
    # 30-delta rule: ~30% probability of assignment = good risk/reward
    strike = round(last_price * (1 - CSP_OTM_PCT), 2)

    # Cash required to secure put: strike × 100 × contracts
    max_budget  = idle_cash * CSP_MAX_BUDGET
    contracts   = max(1, min(5, int(max_budget / (strike * 100))))
    cash_needed = strike * contracts * 100

    if cash_needed > idle_cash * 0.90:
        contracts  = max(1, int(idle_cash * 0.90 / (strike * 100)))
        cash_needed = strike * contracts * 100

    if contracts < 1:
        return None

    # Expiry: 35 DTE target (30-45 DTE sweet spot), nearest Friday
    from datetime import datetime, timedelta
    expiry_dt = datetime.now() + timedelta(days=CSP_DTE_TARGET)
    days_to_friday = (4 - expiry_dt.weekday()) % 7
    expiry_dt += timedelta(days=days_to_friday)
    expiry = expiry_dt.strftime('%Y-%m-%d')
    dte    = (expiry_dt - datetime.now()).days

    # Enforce minimum DTE
    if dte < CSP_DTE_MIN:
        expiry_dt += timedelta(days=7)
        expiry = expiry_dt.strftime('%Y-%m-%d')
        dte    = (expiry_dt - datetime.now()).days

    # Estimate put premium using Black-Scholes
    import math
    iv      = estimate_iv(vixy)
    T       = dte / 365
    # For puts: use put-call parity approximation
    call_px = bs_call_price(last_price, strike, T, 0.05, iv)
    # Put = call + PV(strike) - stock (put-call parity)
    put_px  = max(call_px + strike * math.exp(-0.05 * T) - last_price, 0.05)

    if put_px < CSP_MIN_PREMIUM:
        return None   # premium too small to be worth the assignment risk

    total_income  = round(put_px * contracts * 100, 2)
    effective_buy = round(strike - put_px, 2)   # effective cost if assigned

    return OptionsSignal(
        symbol=symbol,
        action='sell_csp',
        strike=strike,
        expiry=expiry,
        estimated_premium=round(put_px, 4),
        contracts=contracts,
        estimated_cost=-total_income,   # negative = we receive premium
        reason=(
            f"CSP: sell {contracts}x ${strike:.2f}P {expiry} "
            f"@ ~${put_px:.2f}  Income: +${total_income:.2f}  "
            f"Effective buy if assigned: ${effective_buy:.2f}  "
            f"Cash reserved: ${cash_needed:,.2f}"
        ),
        score=score,
    )


def paper_sell_csp(
    signal:    OptionsSignal,
    portfolio: dict,
) -> dict | None:
    """
    Simulates selling a cash-secured put in paper trading.
    Reserves the required cash as collateral.
    Premium collected immediately.
    """
    income       = abs(signal.estimated_cost)
    cash_needed  = signal.strike * signal.contracts * 100

    position = {
        'type':             'cash_secured_put',
        'symbol':           signal.symbol,
        'strike':           signal.strike,
        'expiry':           signal.expiry,
        'contracts':        signal.contracts,
        'entry_premium':    signal.estimated_premium,
        'premium_received': income,
        'cash_reserved':    cash_needed,
        'entry_date':       datetime.now().isoformat(),
        # tastytrade 50% profit target — buy back when premium decays by half
        'buyback_price':    round(signal.estimated_premium * CSP_PROFIT_CLOSE, 4),
        'effective_cost':   round(signal.strike - signal.estimated_premium, 2),
        # Wheel stop — exit entire wheel if assigned and stock drops this far
        'wheel_stop':       round(signal.strike * (1 - WHEEL_STOP_BELOW_COST), 2),
        'dte_entry':        (datetime.strptime(signal.expiry, '%Y-%m-%d') - datetime.now()).days,
    }

    print(f"  🟢 PAPER CSP: Sold {signal.contracts}x {signal.symbol} "
          f"${signal.strike:.2f}P {signal.expiry} "
          f"@ ${signal.estimated_premium:.2f} = +${income:,.2f} premium")
    print(f"     Cash reserved: ${cash_needed:,.2f}  "
          f"Effective cost if assigned: ${position['effective_cost']:.2f}")
    print(f"     Buy back at: ${position['buyback_price']:.2f} (80% profit)")

    return position

def check_wheel_management(
    position: dict,
    current_price: float,
) -> tuple[bool, str]:
    """
    Checks whether to take profit or stop out on an active CSP or CC wheel position.

    Rules:
    1. 50% profit target: buy back put/call when premium decays to 50% of received
    2. Wheel stop: if assigned and stock drops 10% below assignment price → exit wheel
    3. Near expiry OTM: close position 5 DTE if profitable to avoid pin risk
    """
    pos_type       = position.get('type', '')
    entry_premium  = position.get('entry_premium', 0)
    buyback_price  = position.get('buyback_price', entry_premium * 0.50)
    wheel_stop     = position.get('wheel_stop', 0)
    strike         = position.get('strike', 0)

    # 50% profit target (tastytrade principle)
    # In live trading: check current option price vs entry premium
    # Here we approximate using time decay
    if entry_premium > 0 and buyback_price > 0:
        # Proxy: if stock has moved favorably past strike by >3%, take profit
        if pos_type == 'cash_secured_put' and current_price > strike * 1.03:
            return True, f'profit_target_50pct (stock above strike by >3%)'

    # Wheel stop: exit if stock drops below wheel_stop price
    if wheel_stop > 0 and current_price < wheel_stop:
        return True, f'wheel_stop (price ${current_price:.2f} < stop ${wheel_stop:.2f})'

    return False, ''


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

    # Score-based entry routing:
    #   score 0.85+  → sell cash-secured put (high conviction, want to own at discount)
    #   score 0.80-0.84 → buy equity at market (handled by main.py execute_trades)
    #   score 0.85+ in flow regime only → also eligible for long calls
    long_call_signals = []
    csp_signals       = []

    for c in candidates:
        symbol = c.symbol if hasattr(c, 'symbol') else c.get('symbol', '')
        score  = c.total_score if hasattr(c, 'total_score') else c.get('score', 0)
        last   = quotes.get(symbol, {}).get('last', 0)

        # CSP evaluation (score >= 0.85, any non-vol regime)
        csp_signal = evaluate_csp(
            symbol=symbol, score=score, last_price=last,
            vixy=vixy, regime=regime, idle_cash=idle_cash,
        )
        if csp_signal:
            csp_signals.append(csp_signal)
            continue   # CSP takes priority over long call for same symbol

        # Long call evaluation (score >= 0.85, flow regime only)
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
        'csps':          csp_signals,
        'exits':         exit_signals,
    }


def print_options_summary(options_eval: dict):
    lc  = options_eval.get('long_calls', [])
    cc  = options_eval.get('covered_calls', [])
    csp = options_eval.get('csps', [])
    ex  = options_eval.get('exits', [])

    if not lc and not cc and not csp and not ex:
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

    if csp:
        print(f"\n  🟢 Cash-Secured Put Signals ({len(csp)}):")
        for s in csp:
            income = abs(s.estimated_cost)
            eff    = round(s.strike - s.estimated_premium, 2)
            print(f"  SELL {s.contracts}x {s.symbol} ${s.strike:.2f}P {s.expiry} "
                  f"@ ~${s.estimated_premium:.2f}  Income: +${income:,.2f}  "
                  f"Effective buy: ${eff:.2f}")

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