"""
main.py
Master bot loop. Runs every 4 hours during market hours.
Outside market hours: cash manager + pre-scoring only.
Weekends: watchlist prep only.

Cycle order:
  1. Check market hours
  2. Collect data snapshot
  3. Evaluate regime
  4. Manage cash parking
  5. Risk check universe
  6. Score flow momentum
  7. Execute qualifying trades
  8. Log cycle and sleep
"""

import time
import json
import os
from datetime import datetime, timezone
import pytz

from auth import authenticate
from data_collector import (
    collect_snapshot, get_vix, get_vix_history,
    get_price_history, get_quotes, DEFAULT_UNIVERSE,
)
from regime_engine import evaluate_regime, print_regime_summary
from cash_manager import evaluate_cash, get_parking_trades, print_parking_plan
from risk_manager import run_risk_checks
from flow_momentum import run_scoring_cycle, print_cycle_result
from exit_manager import check_all_positions, print_exit_summary
from paper_trader import (
    load_portfolio, paper_buy, paper_sell,
    print_portfolio_summary, estimate_fees,
)

# ── Config ────────────────────────────────────────────────────────────────────

CYCLE_HOURS       = 4          # run every 4 hours
MARKET_OPEN_H     = 9          # 9:30 ET
MARKET_OPEN_M     = 30
MARKET_CLOSE_H    = 16         # 4:00 ET
ET                = pytz.timezone('America/New_York')

# Position sizing: fraction of idle cash per trade (before regime scalar)
BASE_POSITION_PCT = 0.20       # 20% of idle cash per position max

LOG_FILE          = 'bot_log.jsonl'


# ── Market Hours ──────────────────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:   # Saturday=5, Sunday=6
        return False
    if now.hour < MARKET_OPEN_H:
        return False
    if now.hour == MARKET_OPEN_H and now.minute < MARKET_OPEN_M:
        return False
    if now.hour >= MARKET_CLOSE_H:
        return False
    return True


def is_weekend() -> bool:
    return datetime.now(ET).weekday() >= 5


def next_open_str() -> str:
    now = datetime.now(ET)
    if now.weekday() < 5 and now.hour < MARKET_OPEN_H:
        return f"today at 9:30 AM ET"
    days_ahead = (7 - now.weekday()) % 7 or 1
    if now.weekday() >= 5:
        days_ahead = (7 - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 1
    return f"in ~{days_ahead} day(s)"


# ── Logging ───────────────────────────────────────────────────────────────────

def log_cycle(data: dict):
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(data) + '\n')


# ── Position Sizing ───────────────────────────────────────────────────────────

def calc_position_size(idle_cash: float, regime_scalar: float, score: float) -> float:
    """
    Returns dollar amount to deploy in a single position.
    idle_cash * BASE_POSITION_PCT * regime_scalar * score_boost
    Score boost: score >= 0.90 gets 1.2x, else 1.0x
    """
    score_boost = 1.2 if score >= 0.90 else 1.0
    raw = idle_cash * BASE_POSITION_PCT * regime_scalar * score_boost
    return round(raw, 2)


# ── Trade Executor ────────────────────────────────────────────────────────────

def execute_trades(client, candidates, regime_state, portfolio) -> list:
    """
    Executes buys for qualifying candidates.
    Returns list of trade results.
    """
    results = []
    idle_cash = portfolio.get('cash', 0)

    for candidate in candidates:
        symbol      = candidate.symbol
        score       = candidate.total_score
        direction   = candidate.direction

        if direction == 'bearish' and regime_state.regime != 'crisis':
            print(f"  ⏭  Skipping {symbol} — bearish signal in non-crisis regime")
            continue

        position_dollars = calc_position_size(
            idle_cash,
            regime_state.position_size,
            score,
        )

        # Get current price
        from data_collector import get_quote
        q = get_quote(client, symbol)
        price = q.get('last', 0)
        if price <= 0:
            print(f"  ⚠️  Could not get price for {symbol}, skipping")
            continue

        shares = int(position_dollars / price)
        if shares < 1:
            print(f"  ⚠️  {symbol}: position size ${position_dollars:.0f} too small for 1 share @ ${price:.2f}")
            continue

        trade_value = shares * price
        fees = estimate_fees(trade_value)

        print(f"\n  📈 Entering {symbol}")
        print(f"     Score: {score:.3f}  |  Direction: {direction}")
        print(f"     Size:  {shares} shares @ ${price:.2f} = ${trade_value:,.2f}")
        print(f"     Est. fees: ${fees['total_fees']:.2f}")

        success = paper_buy(client, symbol, shares)
        results.append({
            'symbol':    symbol,
            'shares':    shares,
            'price':     price,
            'score':     score,
            'success':   success,
            'timestamp': datetime.now().isoformat(),
        })

        # Update idle cash for next iteration
        if success:
            idle_cash -= (trade_value + fees['total_fees'])

    return results


# ── Cash Parking Executor ─────────────────────────────────────────────────────

def execute_parking(client, plan, portfolio):
    """
    Executes cash parking trades (GLD, SCHP, VTIP, GDX).
    Only trades if rebalance is needed.
    """
    if not plan.needs_rebalance:
        print("  ✅ Cash parking already at target — no rebalance needed")
        return

    from data_collector import get_quotes
    parking_tickers = [t.ticker for t in plan.targets]
    quotes = get_quotes(client, parking_tickers)
    trades = get_parking_trades(plan, portfolio.get('positions', {}), quotes)

    if not trades:
        print("  ✅ No parking trades needed")
        return

    print(f"  🏦 Executing {len(trades)} parking trade(s)...")
    for tr in trades:
        if tr['action'] == 'buy':
            shares = int(tr['dollars'] / tr['price'])
            if shares >= 1:
                paper_buy(client, tr['ticker'], shares)
        elif tr['action'] == 'sell':
            pos = portfolio.get('positions', {}).get(tr['ticker'], {})
            shares = min(int(tr['dollars'] / tr['price']), pos.get('quantity', 0))
            if shares >= 1:
                paper_sell(client, tr['ticker'], shares)


# ── Single Cycle ──────────────────────────────────────────────────────────────

def run_cycle(client):
    cycle_start = datetime.now()
    print(f"\n{'='*60}")
    print(f"  🤖 BOT CYCLE  —  {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    market_open = is_market_open()
    weekend     = is_weekend()

    print(f"  Market open: {'✅ YES' if market_open else '❌ NO'}")
    if not market_open:
        print(f"  Next open: {next_open_str()}")

    # ── Step 1: Fetch core data ───────────────────────────────────────────────
    print("\n📡 Fetching core data...")
    vixy     = get_vix(client)
    vix_hist = get_vix_history(client, days=30)
    print(f"  VIXY: {vixy:.2f}")

    # ── Step 2: Regime ────────────────────────────────────────────────────────
    regime_state = evaluate_regime(vixy, vix_hist)
    print_regime_summary(regime_state)

    if regime_state.in_pause:
        print("  ⏸  Regime just switched — sitting out this cycle")
        log_cycle({'event': 'regime_pause', 'regime': regime_state.regime,
                   'vixy': vixy, 'timestamp': cycle_start.isoformat()})
        return

    # ── Step 3: Portfolio ─────────────────────────────────────────────────────
    portfolio = load_portfolio()
    print(f"  💵 Cash: ${portfolio['cash']:,.2f}  |  "
          f"Positions: {len(portfolio.get('positions', {}))}")

    # ── Step 4: Cash parking (always runs) ───────────────────────────────────
    print("\n🏦 Evaluating cash parking...")
    spy_hist = get_price_history(client, 'SPY', days=30)
    parking_plan = evaluate_cash(regime_state.regime, portfolio)
    print_parking_plan(parking_plan)
    execute_parking(client, parking_plan, portfolio)
    portfolio = load_portfolio()   # reload after parking trades

    # ── Weekend: watchlist prep only, no trading ──────────────────────────────
    if weekend:
        print("\n📋 Weekend mode — building watchlist for Monday...")
        snapshot = collect_snapshot(client, DEFAULT_UNIVERSE)
        result = run_scoring_cycle(
            universe=DEFAULT_UNIVERSE,
            snapshot=snapshot,
            regime=regime_state.regime,
        )
        print_cycle_result(result)
        print("  📝 Watchlist saved — no trades executed (weekend)")
        log_cycle({
            'event':      'weekend_watchlist',
            'regime':     regime_state.regime,
            'vixy':       vixy,
            'candidates': [c.symbol for c in result.candidates],
            'timestamp':  cycle_start.isoformat(),
        })
        return

    # ── Market closed on a weekday: pre-score only ────────────────────────────
    if not market_open:
        print("\n🌙 After hours — pre-scoring tomorrow's universe...")
        snapshot = collect_snapshot(client, DEFAULT_UNIVERSE)
        result = run_scoring_cycle(
            universe=DEFAULT_UNIVERSE,
            snapshot=snapshot,
            regime=regime_state.regime,
        )
        print_cycle_result(result)
        print("  📝 Pre-score complete — no trades executed (market closed)")
        log_cycle({
            'event':      'prescore',
            'regime':     regime_state.regime,
            'vixy':       vixy,
            'candidates': [c.symbol for c in result.candidates],
            'timestamp':  cycle_start.isoformat(),
        })
        return

    # ── Market open: full trading cycle ──────────────────────────────────────
    print("\n📊 Market open — running full trading cycle...")

    # Collect full snapshot
    snapshot = collect_snapshot(client, DEFAULT_UNIVERSE)

    # Risk check every symbol
    print("\n🛡️  Running risk checks...")
    passed_symbols = []
    for symbol in DEFAULT_UNIVERSE:
        result = run_risk_checks(
            symbol=symbol,
            portfolio=portfolio,
            spy_history_df=spy_hist,
            regime=regime_state.regime,
        )
        if result.passed:
            passed_symbols.append(symbol)
        else:
            print(f"  🚫 {symbol} blocked: {', '.join(result.blockers)}")

    print(f"  ✅ {len(passed_symbols)}/{len(DEFAULT_UNIVERSE)} symbols passed risk checks")

    if not passed_symbols:
        print("  ⏸  All symbols blocked — no trades this cycle")
        log_cycle({'event': 'all_blocked', 'regime': regime_state.regime,
                   'timestamp': cycle_start.isoformat()})
        return

    # ── Step: Check exits on existing positions ─────────────────────────────
    print("\n🚪 Checking exit conditions...")
    all_quotes = snapshot.get('quotes', {})
    portfolio  = load_portfolio()
    exit_signals = check_all_positions(portfolio, all_quotes, regime=regime_state.regime)
    print_exit_summary(exit_signals)

    for sig in exit_signals:
        if sig.should_exit:
            pos = portfolio.get('positions', {}).get(sig.symbol, {})
            qty = pos.get('quantity', 0)
            if qty > 0:
                print(f"  🚨 Exiting {sig.symbol} — {sig.reason} "
                      f"(P&L: {sig.current_pnl_pct:+.2f}%)")
                paper_sell(client, sig.symbol, qty)

    portfolio = load_portfolio()   # reload after exits

    # Score passing symbols
    print("\n🔍 Running flow momentum scorer...")
    score_result = run_scoring_cycle(
        universe=passed_symbols,
        snapshot=snapshot,
        regime=regime_state.regime,
    )
    print_cycle_result(score_result)

    # Execute trades
    if score_result.candidates:
        print(f"\n💰 Executing {len(score_result.candidates)} trade(s)...")
        portfolio = load_portfolio()
        trade_results = execute_trades(
            client=client,
            candidates=score_result.candidates,
            regime_state=regime_state,
            portfolio=portfolio,
        )
    else:
        print("\n⏸  No qualifying trades — cash stays parked")
        trade_results = []

    # Final portfolio snapshot
    print("\n📄 End-of-cycle portfolio:")
    print_portfolio_summary(client)

    # Log cycle
    log_cycle({
        'event':         'trading_cycle',
        'regime':        regime_state.regime,
        'vixy':          vixy,
        'position_size': regime_state.position_size,
        'passed_risk':   len(passed_symbols),
        'candidates':    [c.symbol for c in score_result.candidates],
        'trades':        trade_results,
        'timestamp':     cycle_start.isoformat(),
    })


# ── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    print("🤖 Trading bot starting up...")
    print(f"   Cycle interval:   {CYCLE_HOURS}h")
    print(f"   Min score:        0.75")
    print(f"   Max candidates:   4")
    print(f"   Base position:    {BASE_POSITION_PCT*100:.0f}% of idle cash")
    print(f"   Log file:         {LOG_FILE}")

    print("\n🔌 Authenticating with Schwab...")
    client, paper = authenticate()
    mode = "PAPER" if paper else "LIVE"
    print(f"✅ Connected ({mode} mode)\n")

    if not paper:
        confirm = input("⚠️  WARNING: LIVE TRADING MODE. Type 'CONFIRM' to proceed: ")
        if confirm != 'CONFIRM':
            print("Aborted.")
            return

    cycle_count = 0
    while True:
        cycle_count += 1
        print(f"\n{'─'*60}")
        print(f"  Cycle #{cycle_count}")
        try:
            run_cycle(client)
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user.")
            break
        except Exception as e:
            print(f"\n❌ Cycle error: {e}")
            import traceback
            traceback.print_exc()
            log_cycle({'event': 'error', 'error': str(e),
                       'timestamp': datetime.now().isoformat()})

        sleep_seconds = CYCLE_HOURS * 3600
        next_run = datetime.now(ET).strftime('%H:%M:%S')
        print(f"\n💤 Sleeping {CYCLE_HOURS}h — next cycle at "
              f"{(datetime.now(ET).replace(hour=datetime.now(ET).hour)).strftime('%H:%M')} ET")
        print("   (Press Ctrl+C to stop)")

        try:
            time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user.")
            break


if __name__ == '__main__':
    main()