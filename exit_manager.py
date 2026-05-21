"""
exit_manager.py
Monitors all open positions and triggers exits based on three rules:

  1. Stop loss   — exit if position drops > STOP_LOSS_PCT (7%)
  2. Take profit — exit if position gains > TAKE_PROFIT_PCT (15%)
  3. Time stop   — exit after MAX_HOLD_DAYS (5) regardless of P&L

Crisis regime tightens all thresholds:
  Stop loss:   5% (tighter)
  Take profit: 8% (lock in gains faster)
  Max hold:    3 days
"""

import json
import os
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

# ── Thresholds ────────────────────────────────────────────────────────────────

STOP_LOSS_PCT    = 0.05   # 5% base stop loss
TAKE_PROFIT_PCT  = 0.15   # 15% base take profit
MAX_HOLD_DAYS    = 5

CRISIS_STOP_PCT    = 0.05  # 5%
CRISIS_PROFIT_PCT  = 0.08  # 8% -- lock in fast during crisis
CRISIS_HOLD_DAYS   = 3
VOLATILITY_HOLD_DAYS = 5   # same as base but explicit


def dynamic_take_profit(regime: str, adx: float = 0.0) -> float:
    """
    Dynamic take profit target based on regime and trend strength (ADX).

    Logic:
      Crisis:                    8%  -- lock in gains fast, market unstable
      Volatility:               12%  -- slightly looser, more room to run
      Neutral:                  15%  -- standard default
      Flow (ADX < 25):          18%  -- flow regime, moderate trend
      Flow (25 <= ADX < 35):    22%  -- flow regime, strong trend -- let winners run
      Flow (ADX >= 35):         27%  -- flow regime, powerful trend -- maximum extension

    This prevents premature exits in strong uptrends while protecting gains
    in choppy or stressed markets.
    """
    if regime == 'crisis':
        return 0.08
    if regime in ('volatility', 'volatility-cautious', 'volatility-defensive'):
        return 0.12
    if regime == 'flow':
        if adx >= 35:
            return 0.27
        elif adx >= 25:
            return 0.22
        else:
            return 0.18
    # neutral
    return 0.15


# ── Result Container ──────────────────────────────────────────────────────────

@dataclass
class ExitSignal:
    symbol:      str
    should_exit: bool
    reason:      str        # 'stop_loss' | 'take_profit' | 'time_stop' | 'hold'
    current_pnl_pct: float
    days_held:   float
    current_price: float
    avg_price:   float


# ── Core Logic ────────────────────────────────────────────────────────────────

def check_position_exit(
    symbol:        str,
    position:      dict,
    current_price: float,
    regime:        str = 'neutral',
    trade_log:     list = None,
    adx:           float = 0.0,
) -> ExitSignal:
    """
    Checks a single position against all three exit rules.

    Args:
        symbol:        Ticker symbol
        position:      Position dict from portfolio (quantity, avg_price, etc.)
        current_price: Latest market price
        regime:        Current regime string
        trade_log:     Full trade log to find entry timestamp

    Returns:
        ExitSignal with should_exit=True if any rule fires
    """
    crisis      = (regime == 'crisis')
    volatility  = (regime == 'volatility')

    stop_pct   = CRISIS_STOP_PCT      if crisis     else STOP_LOSS_PCT
    profit_pct = dynamic_take_profit(regime, adx=adx)
    max_days   = CRISIS_HOLD_DAYS     if crisis     else                  VOLATILITY_HOLD_DAYS if volatility else MAX_HOLD_DAYS

    avg_price = position.get('avg_price', current_price)
    if avg_price <= 0:
        return ExitSignal(
            symbol=symbol, should_exit=False, reason='hold',
            current_pnl_pct=0.0, days_held=0.0,
            current_price=current_price, avg_price=avg_price,
        )

    pnl_pct = (current_price - avg_price) / avg_price

    # Find entry timestamp from trade log
    days_held = 0.0
    if trade_log:
        entries = [
            t for t in trade_log
            if t.get('symbol') == symbol and t.get('type') == 'BUY'
        ]
        if entries:
            latest_entry = entries[-1]
            entry_time_str = latest_entry.get('timestamp', '')
            try:
                entry_dt = datetime.fromisoformat(entry_time_str)
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                days_held = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 86400
            except Exception:
                days_held = 0.0

    # Rule 1: Stop loss
    if pnl_pct <= -stop_pct:
        return ExitSignal(
            symbol=symbol,
            should_exit=True,
            reason='stop_loss',
            current_pnl_pct=round(pnl_pct * 100, 2),
            days_held=round(days_held, 1),
            current_price=current_price,
            avg_price=avg_price,
        )

    # Rule 2: Take profit
    if pnl_pct >= profit_pct:
        return ExitSignal(
            symbol=symbol,
            should_exit=True,
            reason='take_profit',
            current_pnl_pct=round(pnl_pct * 100, 2),
            days_held=round(days_held, 1),
            current_price=current_price,
            avg_price=avg_price,
        )

    # Rule 3: Time stop
    if days_held >= max_days:
        return ExitSignal(
            symbol=symbol,
            should_exit=True,
            reason='time_stop',
            current_pnl_pct=round(pnl_pct * 100, 2),
            days_held=round(days_held, 1),
            current_price=current_price,
            avg_price=avg_price,
        )

    return ExitSignal(
        symbol=symbol,
        should_exit=False,
        reason='hold',
        current_pnl_pct=round(pnl_pct * 100, 2),
        days_held=round(days_held, 1),
        current_price=current_price,
        avg_price=avg_price,
    )


def check_all_positions(
    portfolio:  dict,
    quotes:     dict,
    regime:     str = 'neutral',
    adx:        float = 0.0,
) -> list[ExitSignal]:
    """
    Checks every open position against exit rules.
    Returns list of ExitSignal objects.
    """
    positions  = portfolio.get('positions', {})
    trade_log  = portfolio.get('trade_log', [])
    signals    = []

    for symbol, position in positions.items():
        # Skip parking positions — never stop-loss GLD/SCHP/VTIP/GDX
        if symbol in ('GLD', 'GDX', 'SCHP', 'VTIP', 'TIP'):
            continue

        price = quotes.get(symbol, {}).get('last', 0)
        if price <= 0:
            price = position.get('avg_price', 0)

        signal = check_position_exit(
            symbol=symbol,
            position=position,
            current_price=price,
            regime=regime,
            trade_log=trade_log,
            adx=adx,
        )
        signals.append(signal)

    return signals


def print_exit_summary(signals: list[ExitSignal]):
    if not signals:
        return
    print(f"\n{'='*52}")
    print(f"  EXIT MANAGER")
    print(f"{'='*52}")
    for s in signals:
        icon = '🚨' if s.should_exit else '✅'
        reason = s.reason.upper().replace('_', ' ')
        print(f"  {icon} {s.symbol:<8} P&L: {s.current_pnl_pct:+.2f}%  "
              f"Held: {s.days_held:.1f}d  [{reason}]")
    print(f"{'='*52}")


# ── Smoke Test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from auth import authenticate
    from data_collector import get_quotes
    from paper_trader import load_portfolio

    print("🔌 Authenticating...")
    client = authenticate()
    paper = True  # assume paper mode
    print(f"✅ Connected ({'PAPER' if paper else 'LIVE'} mode)\n")

    portfolio = load_portfolio()
    positions = portfolio.get('positions', {})

    if not positions:
        print("No open positions to check.")
        print("\nSimulating scenarios:")
        scenarios = [
            ('AAPL', 150.00, 139.00, 'neutral'),   # stop loss
            ('NVDA', 800.00, 920.00, 'neutral'),   # take profit
            ('MSFT', 400.00, 398.00, 'neutral'),   # hold
            ('SPY',  500.00, 470.00, 'crisis'),    # crisis stop loss
        ]
        for sym, avg, curr, regime in scenarios:
            mock_pos = {'avg_price': avg, 'quantity': 10}
            sig = check_position_exit(sym, mock_pos, curr, regime)
            icon = '🚨' if sig.should_exit else '✅'
            print(f"  {icon} {sym:<6} avg=${avg}  curr=${curr}  "
                  f"pnl={sig.current_pnl_pct:+.1f}%  [{sig.reason}]  regime={regime}")
    else:
        syms   = list(positions.keys())
        quotes = get_quotes(client, syms)
        signals = check_all_positions(portfolio, quotes, regime='neutral')
        print_exit_summary(signals)

    print("\n✅ exit_manager.py working correctly.")