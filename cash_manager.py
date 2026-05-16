"""
cash_manager.py
Manages idle cash that is not deployed in active flow/volatility trades.
Instead of sitting flat, cash is parked in inflation-hedging assets
based on the current regime and a persistent high-inflation outlook.

Inflation thesis: High inflation expected for next 3 quarters.
Strategy: Rotate idle cash into TIPS and gold assets — never sit in raw cash.

Cash Parking Rules by Regime:
  flow       →  80% SCHP (TIPS) + 20% GLD  — stay liquid but inflation-protected
  neutral    →  50% SCHP + 30% GLD + 20% VTIP  — balanced hedge
  volatility →  40% SCHP + 40% GLD + 20% VTIP  — heavier gold on vol spike
  crisis     →  35% GLD + 35% GDX + 30% SCHP   — max hard asset, gold miners amplify

Max cash allocation to parking: 90% of idle cash (keep 10% truly liquid for
fast entries). Positions are sized as a % of idle_cash, not total portfolio.

Rebalance trigger: regime change OR allocation drift > 10% from target.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import os

# ── Parking Allocations by Regime ───────────────────────────────────────────

CASH_PARKING = {
    'flow': [
        {'ticker': 'SCHP', 'pct': 0.80, 'note': 'Broad TIPS — inflation hedge'},
        {'ticker': 'GLD',  'pct': 0.20, 'note': 'Gold ETF — hard asset anchor'},
    ],
    'neutral': [
        {'ticker': 'SCHP', 'pct': 0.50, 'note': 'Broad TIPS — core inflation hedge'},
        {'ticker': 'GLD',  'pct': 0.30, 'note': 'Gold ETF — hard asset'},
        {'ticker': 'VTIP', 'pct': 0.20, 'note': 'Short-duration TIPS — less rate risk'},
    ],
    'volatility': [
        {'ticker': 'GLD',  'pct': 0.40, 'note': 'Gold ETF — vol + inflation hedge'},
        {'ticker': 'SCHP', 'pct': 0.40, 'note': 'TIPS — inflation protection'},
        {'ticker': 'VTIP', 'pct': 0.20, 'note': 'Short TIPS — rate buffer'},
    ],
    'crisis': [
        {'ticker': 'GLD',  'pct': 0.35, 'note': 'Gold ETF — crisis hard asset'},
        {'ticker': 'GDX',  'pct': 0.35, 'note': 'Gold miners ETF — amplified gold exposure'},
        {'ticker': 'SCHP', 'pct': 0.30, 'note': 'TIPS — inflation floor'},
    ],
}

LIQUID_RESERVE_PCT = 0.10   # always keep 10% of idle cash truly liquid
MAX_DEPLOY_PCT     = 0.90   # deploy at most 90% of idle cash into parking
REBALANCE_DRIFT    = 0.10   # rebalance if any position drifts > 10% from target


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class ParkingTarget:
    ticker: str
    target_pct: float       # target allocation as fraction of deployable cash
    target_dollars: float   # dollar amount to hold
    note: str


@dataclass
class ParkingPlan:
    regime: str
    idle_cash: float
    deployable_cash: float          # idle_cash * MAX_DEPLOY_PCT
    liquid_reserve: float           # idle_cash * LIQUID_RESERVE_PCT
    targets: list = field(default_factory=list)   # list of ParkingTarget
    needs_rebalance: bool = False
    rebalance_reason: str = ''


# ── Core Logic ────────────────────────────────────────────────────────────────

def compute_idle_cash(portfolio: dict) -> float:
    """
    Returns cash not currently deployed in active positions.
    """
    return float(portfolio.get('cash', 0))


def build_parking_plan(regime: str, idle_cash: float) -> ParkingPlan:
    """
    Builds a target allocation plan for idle cash based on current regime.
    Does NOT execute trades — returns instructions for the bot loop.
    """
    allocations = CASH_PARKING.get(regime, CASH_PARKING['neutral'])
    deployable = idle_cash * MAX_DEPLOY_PCT
    reserve = idle_cash * LIQUID_RESERVE_PCT

    targets = []
    for alloc in allocations:
        dollar_amount = deployable * alloc['pct']
        targets.append(ParkingTarget(
            ticker=alloc['ticker'],
            target_pct=alloc['pct'],
            target_dollars=round(dollar_amount, 2),
            note=alloc['note'],
        ))

    return ParkingPlan(
        regime=regime,
        idle_cash=round(idle_cash, 2),
        deployable_cash=round(deployable, 2),
        liquid_reserve=round(reserve, 2),
        targets=targets,
    )


def check_rebalance_needed(
    plan: ParkingPlan,
    current_positions: dict,
    quotes: dict,
) -> ParkingPlan:
    """
    Checks if current parking positions have drifted from targets.
    Updates plan.needs_rebalance and plan.rebalance_reason.
    current_positions: portfolio['positions'] dict
    quotes: {ticker: quote_dict} from data_collector.get_quotes
    """
    parking_tickers = {t.ticker for t in plan.targets}
    total_parking_value = 0.0

    position_values = {}
    for ticker in parking_tickers:
        pos = current_positions.get(ticker, {})
        qty = pos.get('quantity', 0)
        price = quotes.get(ticker, {}).get('last', pos.get('avg_price', 0))
        val = qty * price
        position_values[ticker] = val
        total_parking_value += val

    if total_parking_value == 0:
        plan.needs_rebalance = True
        plan.rebalance_reason = 'No parking positions found — initial deployment needed'
        return plan

    base = plan.deployable_cash
    for target in plan.targets:
        current_val = position_values.get(target.ticker, 0)
        current_pct = current_val / base if base > 0 else 0
        drift = abs(current_pct - target.target_pct)
        if drift > REBALANCE_DRIFT:
            plan.needs_rebalance = True
            plan.rebalance_reason = (
                f'{target.ticker} drifted {drift*100:.1f}% from target '
                f'({current_pct*100:.1f}% vs {target.target_pct*100:.1f}%)'
            )
            return plan

    return plan


def get_parking_trades(
    plan: ParkingPlan,
    current_positions: dict,
    quotes: dict,
) -> list:
    """
    Returns a list of trade instructions to bring parking positions to target.
    Format: [{'action': 'buy'|'sell', 'ticker': str, 'dollars': float, 'note': str}]
    These are passed to the paper_trader or live order executor.
    """
    trades = []
    for target in plan.targets:
        pos = current_positions.get(target.ticker, {})
        qty = pos.get('quantity', 0)
        price = quotes.get(target.ticker, {}).get('last', pos.get('avg_price', 1))
        if price <= 0:
            continue
        current_val = qty * price
        delta = target.target_dollars - current_val

        if abs(delta) < 50:   # ignore tiny adjustments under $50
            continue

        action = 'buy' if delta > 0 else 'sell'
        trades.append({
            'action':  action,
            'ticker':  target.ticker,
            'dollars': round(abs(delta), 2),
            'shares':  round(abs(delta) / price, 4),
            'price':   round(price, 4),
            'note':    target.note,
        })

    return trades


def print_parking_plan(plan: ParkingPlan):
    icons = {'flow': '🟢', 'neutral': '🟡', 'volatility': '🔴', 'crisis': '🚨'}
    icon = icons.get(plan.regime, '⚪')
    print(f"\n{'='*55}")
    print(f"  CASH MANAGER  {icon} {plan.regime.upper()} REGIME")
    print(f"{'='*55}")
    print(f"  Idle cash:        ${plan.idle_cash:>10,.2f}")
    print(f"  Liquid reserve:   ${plan.liquid_reserve:>10,.2f}  (10% held back)")
    print(f"  Deployable:       ${plan.deployable_cash:>10,.2f}")
    print(f"\n  Target Parking Allocations:")
    print(f"  {'Ticker':<8} {'Allocation':>10} {'$ Target':>12}  Note")
    print(f"  {'-'*52}")
    for t in plan.targets:
        print(f"  {t.ticker:<8} {t.target_pct*100:>9.0f}%  ${t.target_dollars:>10,.2f}  {t.note}")
    if plan.needs_rebalance:
        print(f"\n  ⚠️  Rebalance needed: {plan.rebalance_reason}")
    else:
        print(f"\n  ✅ Allocations within tolerance")
    print(f"{'='*55}\n")


# ── Convenience: Full Cash Management Evaluation ─────────────────────────────

def evaluate_cash(regime: str, portfolio: dict, quotes: dict = None) -> ParkingPlan:
    """
    Top-level call for the bot loop. Returns a fully evaluated ParkingPlan.
    Always checks for missing parking positions.
    If quotes are provided, also checks for drift from targets.
    """
    idle_cash         = compute_idle_cash(portfolio)
    plan              = build_parking_plan(regime, idle_cash)
    current_positions = portfolio.get('positions', {})
    parking_tickers   = {t.ticker for t in plan.targets}

    # Always flag if ANY parking position is completely missing
    missing = [t for t in parking_tickers if t not in current_positions
               or current_positions[t].get('quantity', 0) == 0]
    if missing and idle_cash > 0:
        plan.needs_rebalance  = True
        plan.rebalance_reason = f'Missing parking positions: {", ".join(missing)}'
        return plan

    # If quotes provided, check for drift
    if quotes and idle_cash > 0:
        plan = check_rebalance_needed(plan, current_positions, quotes)

    return plan


# ── Smoke Test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from auth import authenticate
    from data_collector import get_vix, get_vix_history, get_quotes
    from regime_engine import evaluate_regime
    import json

    print("🔌 Authenticating...")
    client, paper = authenticate()
    print(f"✅ Connected ({'PAPER' if paper else 'LIVE'} mode)\n")

    # Load portfolio
    portfolio = {}
    if os.path.exists('paper_portfolio.json'):
        with open('paper_portfolio.json') as f:
            portfolio = json.load(f)
    else:
        portfolio = {'starting_cash': 25000, 'cash': 25000, 'positions': {}}

    idle_cash = compute_idle_cash(portfolio)
    print(f"💵 Idle cash: ${idle_cash:,.2f}")

    # Get current regime
    vixy = get_vix(client)
    vix_history = get_vix_history(client, days=30)
    regime_state = evaluate_regime(vixy, vix_history)
    print(f"📊 Current regime: {regime_state.regime.upper()} (VIXY {vixy:.2f})\n")

    # Fetch quotes for parking instruments
    parking_tickers = list({t for allocs in CASH_PARKING.values() for a in allocs for t in [a['ticker']]})
    print(f"📡 Fetching quotes for: {parking_tickers}")
    quotes = get_quotes(client, parking_tickers)
    for t, q in quotes.items():
        print(f"  {t}: ${q.get('last', 0):.2f}")

    # Build and display plan for current regime
    plan = evaluate_cash(regime_state.regime, portfolio, quotes)
    print_parking_plan(plan)

    # Show all 4 regime plans for comparison
    print("\n📋 All regime parking plans:")
    for regime in ['flow', 'neutral', 'volatility', 'crisis']:
        p = build_parking_plan(regime, idle_cash)
        icons = {'flow': '🟢', 'neutral': '🟡', 'volatility': '🔴', 'crisis': '🚨'}
        print(f"\n  {icons[regime]} {regime.upper()}")
        for t in p.targets:
            print(f"    {t.ticker:<6} {t.target_pct*100:.0f}%  ${t.target_dollars:,.2f}")

    # Show what trades would be needed right now
    trades = get_parking_trades(plan, portfolio.get('positions', {}), quotes)
    if trades:
        print(f"\n🔄 Trades needed to reach target allocation:")
        for tr in trades:
            print(f"  {tr['action'].upper():4} {tr['ticker']:6} {tr['shares']:.2f} shares @ ${tr['price']:.2f}  (${tr['dollars']:,.2f})")
    else:
        print("\n✅ No parking trades needed — already at target.")