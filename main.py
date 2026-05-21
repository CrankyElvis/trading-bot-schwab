"""
main.py
Master bot loop. Runs every 4 hours during market hours.
Outside market hours: cash manager + pre-scoring only.
Weekends: watchlist prep only.

Cycle order:
  1. Check market hours
  2. Fetch VIX, term structure, regime
  3. Portfolio load
  4. Cash parking
  5. Risk checks (9 blockers including volatility regime + circuit breaker)
  6. Exit checks on existing positions
  7. Watchlist candidates or live scoring
  8. Execute equity trades + options signals
  9. Log cycle and sleep
"""

import time
import json
import pandas as pd
import os
from datetime import datetime, timezone, timedelta
import pytz

from auth import authenticate
from data_collector import (
    collect_snapshot, get_vix, get_vix_history,
    get_vix_term_structure, get_price_history,
    get_quotes, DEFAULT_UNIVERSE, get_dp_thresholds_bulk,
    get_politician_tickers, get_wsb_tickers,
)
from regime_engine import evaluate_regime
from macro_sentinel import evaluate_macro, print_macro_report
from notifier import run_anomaly_checks, send_daily_summary
from cash_manager import evaluate_cash, get_parking_trades, print_parking_plan
from risk_manager import run_risk_checks
from flow_momentum import run_scoring_cycle, print_cycle_result, StockScore
from pre_market_scanner import load_watchlist, get_watchlist_candidates
from signal_exit_manager import (
    check_signal_exits, print_signal_exit_summary,
    enrich_position_metadata,
)
from options_manager import evaluate_options, print_options_summary, paper_buy_call, paper_sell_covered_call
from paper_trader import (
    load_portfolio, paper_buy, paper_sell,
    print_portfolio_summary, estimate_fees,
)

# ── Config ────────────────────────────────────────────────────────────────────

MAX_CANDIDATES    = 4
BASE_POSITION_PCT = 0.20

# ── Top-of-funnel source sizing overrides ─────────────────────────────────────
POLITICIAN_POSITION_PCT = 0.10   # 10% — liquidity risk on non-standard symbols
WSB_POSITION_PCT        = 0.08   # 8%  — volatility risk on meme names
POLITICIAN_STOP_PCT     = 0.07   # wider stop — hold longer, momentum fades slowly
WSB_STOP_PCT            = 0.03   # tight stop — fast reversal risk
POLITICIAN_TP_PCT       = 0.25   # flat 25% take profit
WSB_TP_PCT              = 0.20   # flat 20% take profit — take fast
POLITICIAN_MAX_HOLD     = 10     # days
WSB_MAX_HOLD            = 2      # days
LOG_FILE          = 'bot_log.jsonl'
ET                = pytz.timezone('America/New_York')

# ── Named Trading Schedule (ET) ───────────────────────────────────────────────
# Each entry: (hour, minute, cycle_name, action)
# Actions: 'premarket_scan' | 'trade' | 'exit_only' | 'after_hours'
SCHEDULE = [
    ( 6,  0, 'premarket',    'premarket_scan'),  # 6:00am — full 141-symbol scan
    ( 9, 25, 'pre_open',     'trade'),            # 9:25am — load watchlist, ready to fire
    ( 9, 35, 'open',         'trade'),            # 9:35am — first settled cycle
    (11, 30, 'mid_morning',  'trade'),            # 11:30am — late morning momentum
    (13, 30, 'midday',       'trade'),            # 1:30pm  — post-lunch repositioning
    (15,  0, 'power_hour',   'trade'),            # 3:00pm  — institutional end-of-day flow
    (16,  5, 'close',        'exit_only'),        # 4:05pm  — exit checks, CC evaluation
    (18,  0, 'after_hours',  'after_hours'),      # 6:00pm  — log P&L, overnight prep
]

MARKET_OPEN_H  = 9
MARKET_OPEN_M  = 30
MARKET_CLOSE_H = 16


# ── Market Hours ──────────────────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
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
        return "today at 9:30 AM ET"
    if now.weekday() >= 5:
        days_ahead = (7 - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 1
    else:
        days_ahead = 1
    return f"in ~{days_ahead} day(s)"


def get_next_cycle() -> tuple:
    """
    Returns (seconds_until_next, cycle_name, action) for the next scheduled cycle.
    On weekends, sleeps until Monday 6:00am ET.
    """
    now = datetime.now(ET)

    # Weekend — sleep until Monday 6am
    if now.weekday() >= 5:
        days_until_monday = (7 - now.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7
        monday_6am = now.replace(hour=6, minute=0, second=0, microsecond=0)
        monday_6am += timedelta(days=days_until_monday)
        wait = (monday_6am - now).total_seconds()
        return wait, 'monday_open', 'premarket_scan'

    # Weekday — find next scheduled slot
    today_slots = []
    for h, m, name, action in SCHEDULE:
        slot_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if slot_dt > now:
            today_slots.append((slot_dt, name, action))

    if today_slots:
        next_dt, name, action = today_slots[0]
        wait = (next_dt - now).total_seconds()
        return wait, name, action

    # All slots passed today — wait until tomorrow 6am
    tomorrow_6am = (now + timedelta(days=1)).replace(
        hour=6, minute=0, second=0, microsecond=0
    )
    # Skip to Monday if tomorrow is weekend
    while tomorrow_6am.weekday() >= 5:
        tomorrow_6am += timedelta(days=1)
    wait = (tomorrow_6am - now).total_seconds()
    return wait, 'premarket', 'premarket_scan'


# Track which cycles have already fired today
_cycles_fired: set = set()

def get_current_cycle() -> tuple:
    """
    Returns (cycle_name, action) for the current moment.
    Each named cycle only fires once per calendar day.
    Returns ('sleep', 'sleep') if the current cycle already fired today.
    """
    now = datetime.now(ET)

    if now.weekday() >= 5:
        return 'weekend', 'weekend'

    # Find most recent past slot
    last_name, last_action = 'overnight', 'sleep'
    for h, m, name, action in SCHEDULE:
        slot_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if slot_dt <= now:
            last_name, last_action = name, action

    # Gate: each cycle fires once per calendar day
    today = now.strftime('%Y-%m-%d')
    cycle_key = f"{today}_{last_name}"
    if cycle_key in _cycles_fired:
        return 'sleep', 'sleep'

    _cycles_fired.add(cycle_key)
    # Clean up old keys (keep only today's)
    old_keys = {k for k in _cycles_fired if not k.startswith(today)}
    _cycles_fired.difference_update(old_keys)

    return last_name, last_action


# ── Logging ───────────────────────────────────────────────────────────────────

def log_cycle(data: dict):
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(data) + '\n')


# ── Position Sizing ───────────────────────────────────────────────────────────

def calc_position_size(idle_cash: float, regime_scalar: float, score: float) -> float:
    score_boost = 1.2 if score >= 0.90 else 1.0
    return round(idle_cash * BASE_POSITION_PCT * regime_scalar * score_boost, 2)


# ── Trade Executor ────────────────────────────────────────────────────────────

def execute_trades(client, candidates, regime_state, portfolio) -> list:
    results   = []
    idle_cash = portfolio.get('cash', 0)

    for candidate in candidates:
        symbol    = candidate.symbol
        score     = candidate.total_score
        direction = candidate.direction

        if direction == 'bearish' and regime_state.regime != 'crisis':
            print(f"  ⏭  Skipping {symbol} — bearish signal in non-crisis regime")
            continue

        position_dollars = calc_position_size(idle_cash, regime_state.position_size, score)

        from data_collector import get_quote
        q     = get_quote(client, symbol)
        price = q.get('last', 0)
        if price <= 0:
            print(f"  ⚠️  Could not get price for {symbol}, skipping")
            continue

        shares = int(position_dollars / price)
        if shares < 1:
            print(f"  ⚠️  {symbol}: ${position_dollars:.0f} too small for 1 share @ ${price:.2f}")
            continue

        trade_value = shares * price
        fees        = estimate_fees(trade_value)

        print(f"\n  📈 Entering {symbol}")
        print(f"     Score: {score:.3f}  |  Direction: {direction}")
        print(f"     Size:  {shares} shares @ ${price:.2f} = ${trade_value:,.2f}")
        print(f"     Est. fees: ${fees['total_fees']:.2f}")

        success = paper_buy(client, symbol, shares)
        if success:
            # Store entry metadata for signal-based exit tracking
            from paper_trader import load_portfolio, save_portfolio
            pf = load_portfolio()
            if symbol in pf.get('positions', {}):
                pf['positions'][symbol] = enrich_position_metadata(
                    position=pf['positions'][symbol],
                    entry_score=score,
                    entry_direction=direction,
                    spy_history=get_price_history(client, 'SPY', days=30),
                    price_history={symbol: get_price_history(client, symbol, days=30)},
                    symbol=symbol,
                )
                save_portfolio(pf)

        results.append({
            'symbol': symbol, 'shares': shares, 'price': price,
            'score': score, 'success': success,
            'timestamp': datetime.now().isoformat(),
        })
        if success:
            idle_cash -= (trade_value + fees['total_fees'])

    return results


# ── Cash Parking Executor ─────────────────────────────────────────────────────

def execute_parking(client, plan, portfolio):
    if not plan.needs_rebalance:
        print("  ✅ Cash parking already at target — no rebalance needed")
        return

    parking_tickers = [t.ticker for t in plan.targets]
    quotes          = get_quotes(client, parking_tickers)
    trades          = get_parking_trades(plan, portfolio.get('positions', {}), quotes)

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
            pos    = portfolio.get('positions', {}).get(tr['ticker'], {})
            shares = min(int(tr['dollars'] / tr['price']), pos.get('quantity', 0))
            if shares >= 1:
                paper_sell(client, tr['ticker'], shares)


# ── Single Cycle ──────────────────────────────────────────────────────────────

def run_cycle(client):
    cycle_start              = datetime.now()
    cycle_name, cycle_action = get_current_cycle()
    if cycle_action == 'sleep':
        time.sleep(60)
        return

    print(f"\n{'='*60}")
    print(f"  🤖 BOT CYCLE [{cycle_name.upper()}]  —  "
          f"{cycle_start.strftime('%Y-%m-%d %H:%M:%S ET')}")
    print(f"{'='*60}")

    market_open = is_market_open()
    weekend     = is_weekend()

    print(f"  Market open: {'✅ YES' if market_open else '❌ NO'}  "
          f"| Cycle: {cycle_name}  | Action: {cycle_action}")
    if not market_open:
        print(f"  Next open: {next_open_str()}")

    # ── Step 1: Core data ────────────────────────────────────────────────────
    print("\n📡 Fetching core data...")
    vixy     = get_vix(client)
    vix_hist = get_vix_history(client, days=30)
    print(f"  VIXY: {vixy:.2f}")

    print("  Fetching VIX term structure...")
    term_structure = get_vix_term_structure()
    ts = term_structure.get
    print(f"  Term structure: VIX9D={ts('vix9d',0):.1f}  "
          f"VIX3M={ts('vix3m',0):.1f}  "
          f"Signal={ts('signal','flat').upper()}"
          f"{'  🛑 HALT ENTRIES' if ts('halt_entries',False) else ''}")

    # ── Step 2: Regime ────────────────────────────────────────────────────────
    regime_state = evaluate_regime(vixy, vix_hist, term_structure)

    if regime_state.in_pause:
        print("  ⏸  Regime just switched — sitting out this cycle")
        log_cycle({'event': 'regime_pause', 'regime': regime_state.regime,
                   'vixy': vixy, 'timestamp': cycle_start.isoformat()})
        return

    # ── Step 3: Portfolio ─────────────────────────────────────────────────────
    portfolio = load_portfolio()
    print(f"  💵 Cash: ${portfolio['cash']:,.2f}  |  "
          f"Positions: {len(portfolio.get('positions', {}))}")

    # ── Step 4: Cash parking — DISABLED pending evaluation ──────────────────
    print("\n🏦 Cash parking: DISABLED (backlog item #2 — pending evaluation)")
    # Re-enable by uncommenting:
    spy_hist         = get_price_history(client, 'SPY', days=30)
    # parking_tickers = ['GLD', 'SCHP', 'VTIP', 'GDX']
    # parking_quotes  = get_quotes(client, parking_tickers)
    # parking_plan    = evaluate_cash(regime_state.regime, portfolio, parking_quotes)
    # print_parking_plan(parking_plan)
    # execute_parking(client, parking_plan, portfolio)
    portfolio = load_portfolio()

    # ── Top-of-funnel universe injection ─────────────────────────────────────
    injected_symbols = {}   # {symbol: source}
    try:
        pol_tickers = get_politician_tickers(days=90, min_buy_count=1)
        for t in pol_tickers:
            sym = t['symbol']
            if sym not in DEFAULT_UNIVERSE:
                injected_symbols[sym] = 'politician'
                print(f"  [top-funnel] POLITICIAN injected: {sym} "
                      f"({t['buy_count']} buy(s))")
    except Exception as e:
        print(f"  [top-funnel] Politician scan error: {e}")

    try:
        wsb_tickers = get_wsb_tickers(min_mentions=5, min_volume_surge=1.5)
        for t in wsb_tickers:
            sym = t['symbol']
            if sym not in DEFAULT_UNIVERSE and sym not in injected_symbols:
                injected_symbols[sym] = 'wsb'
                print(f"  [top-funnel] WSB injected: {sym} "
                      f"({t['mentions']} mentions, {t['volume_surge']}x vol)")
    except Exception as e:
        print(f"  [top-funnel] WSB scan error: {e}")

    cycle_universe = list(DEFAULT_UNIVERSE) + [s for s in injected_symbols
                                                if s not in DEFAULT_UNIVERSE]

    # ── Pre-market scan (6:00am) ─────────────────────────────────────────────
    # ── Close overnight position at open ─────────────────────────────────────
    if cycle_action in ('pre_open', 'open') and not weekend:
        portfolio = load_portfolio()
        _close_overnight_position(client, portfolio)

    if cycle_action == 'premarket_scan' and not weekend:
        print("\n🌅 Pre-market scan — scoring full 141-symbol universe...")

        # ── Macro sentinel -- fresh leading indicator read ─────────────
        try:
            macro = evaluate_macro(use_cache=False)
            print_macro_report(macro)
        except Exception as e:
            print(f'  [macro] Premarket sentinel failed: {e}')

        from pre_market_scanner import run_scan
        run_scan(client)
        log_cycle({'event': 'premarket_scan', 'regime': regime_state.regime,
                   'timestamp': cycle_start.isoformat()})
        return

    # ── Exit-only cycle (4:05pm close) ────────────────────────────────────────
    if cycle_action == 'exit_only':
        print("\n🔔 Close cycle — checking exits and options...")
        portfolio    = load_portfolio()
        all_quotes   = get_quotes(client, list(portfolio.get('positions', {}).keys()))
        exit_results = check_signal_exits(
            portfolio=portfolio, quotes=all_quotes,
            spy_history=get_price_history(client, 'SPY', days=30),
            price_history={}, uw_flow_df=pd.DataFrame(),
            dp_df=pd.DataFrame(), regime=regime_state.regime,
        )
        print_signal_exit_summary(exit_results)
        for sig in exit_results:
            if sig.should_exit:
                pos = portfolio.get('positions', {}).get(sig.symbol, {})
                qty = pos.get('quantity', 0)
                if qty > 0:
                    paper_sell(client, sig.symbol, qty)
        log_cycle({'event': 'exit_only_cycle', 'regime': regime_state.regime,
                   'timestamp': cycle_start.isoformat()})
        return

    # ── After-hours (6:00pm) ──────────────────────────────────────────────────
    if cycle_action == 'after_hours':
        print("\n🌙 After-hours — logging P&L and overnight prep...")
        portfolio = load_portfolio()
        print_portfolio_summary(client)
        log_cycle({'event': 'after_hours', 'regime': regime_state.regime,
                   'timestamp': cycle_start.isoformat()})
        try:
            send_daily_summary(portfolio, regime_state=regime_state)
            print("  📧 Daily summary email sent")
        except Exception as e:
            print(f"  ⚠️  Daily summary email failed: {e}")
        return

    # ── Close cycle overnight index strategy ─────────────────────────────────
    # Runs at 4:05pm ET after exits are processed
    # Buy index ETF at close, sell at next open to capture overnight drift
    if cycle_action == 'exit_only' and not weekend:
        _run_overnight_strategy(client, portfolio, regime_state)

    # ── Weekend: watchlist prep only ──────────────────────────────────────────
    if weekend:
        print("\n📋 Weekend mode — building watchlist for Monday...")
        snapshot = collect_snapshot(client, cycle_universe)
        port_val = portfolio.get('cash', 25000)
        result   = run_scoring_cycle(
            universe=cycle_universe,
            snapshot=snapshot,
            regime=regime_state.regime,
            portfolio_value=port_val,
        )
        print_cycle_result(result)
        print("  📝 Watchlist saved — no trades executed (weekend)")
        log_cycle({
            'event':      'weekend_watchlist',
            'regime':     regime_state.regime,
            'vixy':       vixy,
            'term_signal': term_structure.get('signal', 'flat'),
            'candidates': [c.symbol for c in result.candidates],
            'timestamp':  cycle_start.isoformat(),
        })
        return

    # ── Pre-open or after hours: pre-score only ──────────────────────────────
    if not market_open:
        print("\n🌙 After hours — pre-scoring tomorrow's universe...")
        snapshot = collect_snapshot(client, cycle_universe)
        port_val = portfolio.get('cash', 25000)
        result   = run_scoring_cycle(
            universe=cycle_universe,
            snapshot=snapshot,
            regime=regime_state.regime,
            portfolio_value=port_val,
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
    snapshot = collect_snapshot(client, cycle_universe)

    # Pre-fetch float-adjusted dark pool thresholds for all symbols
    print("  Fetching float-adjusted dark pool thresholds...")
    dp_thresholds = get_dp_thresholds_bulk(client, DEFAULT_UNIVERSE)

    # ── Step 5: Risk checks ───────────────────────────────────────────────────
    print("\n🛡️  Running risk checks...")
    passed_symbols = []
    for symbol in DEFAULT_UNIVERSE:
        result = run_risk_checks(
            symbol=symbol,
            portfolio=portfolio,
            spy_history_df=spy_hist,
            regime=regime_state.regime,
            price_history=snapshot.get('price_history', {}),
            term_structure=term_structure,
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

    # ── Step 6: Signal-based exit checks ────────────────────────────────────
    print("\n🚪 Checking signal-based exit conditions...")
    all_quotes   = snapshot.get('quotes', {})
    portfolio    = load_portfolio()
    price_hist   = snapshot.get('price_history', {})
    uw_flow_frames = [df for df in snapshot.get('uw_flow', {}).values() if not df.empty]
    combined_flow = pd.concat(uw_flow_frames, ignore_index=True) if uw_flow_frames else pd.DataFrame()
    dp_df        = snapshot.get('uw_darkpool', pd.DataFrame())

    exit_results = check_signal_exits(
        portfolio=portfolio,
        quotes=all_quotes,
        spy_history=spy_hist,
        price_history=price_hist,
        uw_flow_df=combined_flow,
        dp_df=dp_df,
        regime=regime_state.regime,
    )
    print_signal_exit_summary(exit_results)

    for sig in exit_results:
        if sig.covered_call_mode:
            # Position is up 15-25% — sell covered call instead of exiting equity
            pos = portfolio.get('positions', {}).get(sig.symbol, {})
            qty = pos.get('quantity', 0)
            if qty >= 100:
                cc_signal = OptionsSignal(
                    symbol=sig.symbol,
                    action='sell_covered_call',
                    strike=sig.cc_strike,
                    expiry=sig.cc_expiry,
                    estimated_premium=sig.cc_est_premium,
                    contracts=qty // 100,
                    estimated_cost=-(sig.cc_est_premium * (qty // 100) * 100),
                    reason=f"Covered call exit — position up {sig.current_pnl_pct:.1f}%",
                )
                print(f"  💰 CC EXIT: {sig.symbol} up {sig.current_pnl_pct:.1f}% "
                      f"→ selling ${sig.cc_strike:.2f}C {sig.cc_expiry} "
                      f"@ ~${sig.cc_est_premium:.2f}")
                paper_sell_covered_call(cc_signal, portfolio)
            else:
                # Not enough shares for CC — hard sell instead
                print(f"  ⚠️  {sig.symbol}: CC mode but <100 shares — hard selling")
                paper_sell(client, sig.symbol, qty)

        elif sig.should_exit:
            pos = portfolio.get('positions', {}).get(sig.symbol, {})
            qty = pos.get('quantity', 0)
            if qty > 0:
                print(f"  🚨 Exiting {sig.symbol} — {sig.reason} "
                      f"(P&L: {sig.current_pnl_pct:+.2f}%  "
                      f"Score: {sig.entry_score:.3f}→{sig.current_score:.3f})")
                paper_sell(client, sig.symbol, qty)

    portfolio = load_portfolio()

    # ── Step 7: Candidates from watchlist or live scoring ─────────────────────
    print("\n🔍 Checking pre-market watchlist...")
    watchlist    = load_watchlist()
    trade_results = []

    if watchlist:
        print(f"  ✅ Watchlist from {watchlist.get('scanned_at','unknown')[:19]}")
        print(f"  Scanned: {watchlist.get('symbols_scanned',0)}  "
              f"Qualified: {watchlist.get('qualified',0)}")

        wl_candidates = get_watchlist_candidates(min_score=0.75, max_n=20)
        wl_candidates = [c for c in wl_candidates if c['symbol'] in passed_symbols]

        if wl_candidates:
            print(f"\n  🎯 {len(wl_candidates)} candidates passed risk checks:")
            for c in wl_candidates[:MAX_CANDIDATES]:
                print(f"    {c['symbol']:<8} score={c['score']:.3f}  "
                      f"dir={c['direction']}  rsi={c.get('rsi',0):.1f}")

            candidates = [
                StockScore(
                    symbol=c['symbol'],
                    total_score=c['score'],
                    signals=c.get('signals', {}),
                    weighted=c.get('weighted', {}),
                    qualifies=True,
                    direction=c['direction'],
                )
                for c in wl_candidates[:MAX_CANDIDATES]
            ]
        else:
            print("  ⏸  No watchlist candidates passed risk checks")
            candidates = []
    else:
        print("  ⚠️  No watchlist — running live scoring...")
        port_val = portfolio.get('cash', 0) + sum(
            p.get('quantity', 0) * p.get('avg_price', 0)
            for p in portfolio.get('positions', {}).values()
        )
        score_result = run_scoring_cycle(
            universe=passed_symbols,
            snapshot=snapshot,
            regime=regime_state.regime,
            portfolio_value=port_val,
        )
        print_cycle_result(score_result)
        candidates = score_result.candidates

    # ── Step 8: Execute equity trades ─────────────────────────────────────────
    if candidates:
        print(f"\n💰 Executing {len(candidates)} equity trade(s)...")
        portfolio     = load_portfolio()
        trade_results = execute_trades(
            client=client,
            candidates=candidates,
            regime_state=regime_state,
            portfolio=portfolio,
        )
    else:
        print("\n⏸  No qualifying equity trades — cash stays parked")

    # ── Step 9: Options signals ────────────────────────────────────────────────
    print("\n🎯 Evaluating options signals...")
    portfolio        = load_portfolio()
    options_eval     = evaluate_options(
        candidates=candidates,
        portfolio=portfolio,
        quotes=all_quotes,
        vixy=vixy,
        regime=regime_state.regime,
    )
    print_options_summary(options_eval)

    # Execute long calls
    for signal in options_eval.get('long_calls', []):
        portfolio = load_portfolio()
        pos = paper_buy_call(signal, portfolio, client)
        if pos:
            print(f"  ✅ Long call entered: {signal.symbol}")

    # Execute covered calls
    for signal in options_eval.get('covered_calls', []):
        portfolio = load_portfolio()
        pos = paper_sell_covered_call(signal, portfolio)
        if pos:
            print(f"  ✅ Covered call sold: {signal.symbol}")

    # Execute cash-secured puts (high-conviction entries at a discount)
    from options_manager import paper_sell_csp
    for signal in options_eval.get('csps', []):
        portfolio = load_portfolio()
        if signal.symbol in portfolio.get('positions', {}):
            continue
        pos = paper_sell_csp(signal, portfolio)
        if pos:
            print(f"  ✅ CSP sold: {signal.symbol} — "
                  f"effective buy ${pos['effective_cost']:.2f} if assigned")

    # ── Volatility Playbook ────────────────────────────────────────────────────
    if regime_state.regime == 'volatility':
        portfolio  = load_portfolio()
        idle_cash  = portfolio.get('cash', 0)
        all_quotes = get_quotes(client, ['SPY','QQQ','XLP','XLU','XLV','GLD','GDX'])

        # 1. Defensive sector rotation
        def_signals = evaluate_defensive_rotation(
            regime=regime_state.regime, vixy=regime_state.vixy,
            quotes=all_quotes, idle_cash=idle_cash,
        )
        for sig in def_signals:
            print(f"  🛡️  DEFENSIVE: BUY {sig['shares']}x {sig['symbol']} "
                  f"@ ${sig['price']:.2f} = ${sig['cost']:,.0f}  ({sig['note']})")

        # 2. VIX mean reversion (SPY calls on spike)
        mr_signal = evaluate_vix_mean_reversion(
            regime=regime_state.regime,
            mean_reversion_signal=regime_state.mean_reversion_signal,
            vixy=regime_state.vixy,
            vix_zscore=regime_state.vix_zscore,
            quotes=all_quotes, idle_cash=idle_cash,
        )
        if mr_signal:
            print(f"  🎯 MEAN REVERSION: {mr_signal['note']}")

        # 3. Iron condor on SPY/QQQ when vol stable
        for ic_sym in ['SPY', 'QQQ']:
            ic = evaluate_iron_condor(
                regime=regime_state.regime, vixy=regime_state.vixy,
                vix_zscore=regime_state.vix_zscore,
                term_signal=regime_state.term_signal,
                quotes=all_quotes, idle_cash=idle_cash, symbol=ic_sym,
            )
            if ic:
                print(f"  📐 IRON CONDOR: {ic['note']}")
                break   # one condor at a time

    # ── Final portfolio snapshot ───────────────────────────────────────────────
    print("\n📄 End-of-cycle portfolio:")
    print_portfolio_summary(client)

    log_cycle({
        'event':         'trading_cycle',
        'regime':        regime_state.regime,
        'term_signal':   term_structure.get('signal', 'flat'),
        'vixy':          vixy,
        'position_size': regime_state.position_size,
        'passed_risk':   len(passed_symbols),
        'candidates':    [c.symbol for c in candidates],
        'trades':        trade_results,
        'timestamp':     cycle_start.isoformat(),
    })


# ── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    print("🤖 Trading bot starting up...")
    print(f"   Schedule:         {len(SCHEDULE)} named cycles/day")
    print(f"   Min score:        0.75")
    print(f"   Max candidates:   {MAX_CANDIDATES}")
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

        wait_sec, next_name, next_action = get_next_cycle()
        next_dt = datetime.now(ET) + timedelta(seconds=wait_sec)
        print(f"\n💤 Next cycle: {next_name.upper()} ({next_action}) "
              f"at {next_dt.strftime('%I:%M %p ET')} "
              f"(in {int(wait_sec//3600)}h {int((wait_sec%3600)//60)}m)")
        print("   (Press Ctrl+C to stop)")

        try:
            time.sleep(wait_sec)
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user.")
            break


if __name__ == '__main__':
    main()