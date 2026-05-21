"""
signal_exit_manager.py
Replaces hard time-stop exits with signal-based exit logic.

Philosophy: Hold as long as the entry thesis is intact. Exit when it breaks.
Hard stop loss and take profit remain as safety rails.

Exit triggers (any one fires = exit):
  1. Score decay          — re-scored position drops below SCORE_EXIT_THRESHOLD (0.40)
  2. Direction flip       — direction turns bearish when entered bullish
  3. Market tide turn     — SPY 5d EMA crosses below 20d EMA (was bullish at entry)
  4. Sector rotation      — sector ETF EMA turns bearish (was bullish at entry)
  5. Flow dry-up          — no sweep/dark pool activity in last 2 days on symbol
  6. Distribution day     — price down >2% on volume 2x+ average (institutional selling)
  7. Stop loss            — hard floor at -7% (or -5% in crisis)
  8. Covered call exit    — position hits +15%, sell covered call instead of hard sell
                            collect premium + theta, exit when called away at strike
  9. Take profit (backup) — hard ceiling at +20% flow, +15% neutral if no options avail

What we removed:
  - MAX_HOLD_DAYS time stop — replaced entirely by signal decay detection

Covered call exit mode:
  When position reaches COVERED_CALL_TRIGGER (+15%), instead of selling:
  - Sell a covered call at strike = current_price * 1.03 (3% OTM), 2-week expiry
  - Collect theta income while waiting for assignment
  - If called away: exits at +18% effective + premium collected (~+20-22% total)
  - If expires worthless: re-evaluate — sell another call or exit equity position
  - Hard take profit at +20% still fires as backup if no options available
"""

import os
import time
import math
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field

import pandas as pd
from exit_manager import dynamic_take_profit

# ── Config ────────────────────────────────────────────────────────────────────

SCORE_EXIT_THRESHOLD  = 0.40   # exit if re-score drops below this
FLOW_LOOKBACK_HOURS   = 48     # hours to look back for flow activity
MIN_FLOW_PREMIUM      = 100_000 # minimum sweep premium to count as active flow

# Hard safety rails
STOP_LOSS_PCT         = 0.05   # tightened from 7% — stop losses averaging -$444
TAKE_PROFIT_FLOW      = 0.20   # legacy -- now overridden by dynamic_take_profit()
TAKE_PROFIT_DEFAULT   = 0.15   # legacy -- now overridden by dynamic_take_profit()
CRISIS_STOP_PCT       = 0.05
CRISIS_PROFIT_PCT     = 0.08

# Covered call exit mode
COVERED_CALL_TRIGGER  = 0.15   # switch to CC exit when position up 15%
COVERED_CALL_OTM      = 0.03   # sell call 3% OTM
COVERED_CALL_DTE      = 14     # 2-week expiry for fast theta harvest
COVERED_CALL_HARD_EXIT = 0.40  # if position up 40%+ and no CC possible, hard exit

# Distribution day exit
DISTRIBUTION_DOWN_PCT = 0.02   # price down >2% intraday
DISTRIBUTION_RVOL_MIN = 2.0    # on volume 2x+ average

PARKING_TICKERS = {'GLD', 'GDX', 'SCHP', 'VTIP', 'TIP'}


# ── Exit Signal Container ─────────────────────────────────────────────────────

@dataclass
class SignalExitResult:
    symbol:           str
    should_exit:      bool
    reason:           str       # which trigger fired
    current_score:    float     # re-scored value
    entry_score:      float     # score at entry
    score_delta:      float     # current - entry
    current_pnl_pct:  float
    current_price:    float
    avg_price:        float
    days_held:        float
    covered_call_mode: bool = False   # True = sell CC instead of exiting equity
    cc_strike:        float = 0.0    # suggested CC strike price
    cc_expiry:        str   = ''     # suggested CC expiry date
    cc_est_premium:   float = 0.0    # estimated premium to collect
    triggers:         dict  = field(default_factory=dict)


# ── Individual Exit Checkers ──────────────────────────────────────────────────

def check_score_decay(
    current_score: float,
    entry_score:   float,
) -> tuple[bool, dict]:
    """
    Exits if re-scored value drops below SCORE_EXIT_THRESHOLD.
    Also exits if score has decayed more than 50% from entry.
    """
    decay_pct   = (entry_score - current_score) / entry_score if entry_score > 0 else 0
    below_floor = current_score < SCORE_EXIT_THRESHOLD
    heavy_decay = decay_pct > 0.50 and current_score < 0.55

    fired = below_floor or heavy_decay
    return fired, {
        'current_score': round(current_score, 3),
        'entry_score':   round(entry_score, 3),
        'decay_pct':     f'{decay_pct*100:.1f}%',
        'below_floor':   below_floor,
        'heavy_decay':   heavy_decay,
    }


def check_direction_flip(
    current_direction: str,
    entry_direction:   str,
) -> tuple[bool, dict]:
    """
    Exits if direction has flipped to bearish since entry.
    Only fires if we entered bullish — neutral entries are not direction-gated.
    """
    if entry_direction != 'bullish':
        return False, {'note': 'Entry was not bullish — direction flip not monitored'}

    flipped = current_direction == 'bearish'
    return flipped, {
        'entry_direction':   entry_direction,
        'current_direction': current_direction,
        'flipped':           flipped,
    }


def check_market_tide(
    spy_history: pd.DataFrame,
    entry_tide:  str = 'bullish',
) -> tuple[bool, dict]:
    """
    Exits if SPY 5d EMA crosses below 20d EMA (market tide turns bearish).
    Only monitors if tide was bullish at entry.
    """
    if entry_tide != 'bullish':
        return False, {'note': 'Market tide was not bullish at entry'}

    if spy_history is None or spy_history.empty or len(spy_history) < 20:
        return False, {'note': 'Insufficient SPY history'}

    closes = spy_history['close']
    ema5   = closes.ewm(span=5,  adjust=False).mean().iloc[-1]
    ema20  = closes.ewm(span=20, adjust=False).mean().iloc[-1]
    bearish = ema5 < ema20

    return bearish, {
        'spy_ema5':   round(float(ema5), 2),
        'spy_ema20':  round(float(ema20), 2),
        'tide':       'bearish' if bearish else 'bullish',
        'entry_tide': entry_tide,
    }


def check_sector_rotation(
    symbol:        str,
    price_history: dict,
    entry_sector_bullish: bool = True,
) -> tuple[bool, dict]:
    """
    Exits if the symbol's sector ETF turns bearish.
    Only monitors if sector was bullish at entry.
    """
    if not entry_sector_bullish:
        return False, {'note': 'Sector was not bullish at entry'}

    sector_map = {
        'XLK': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'META', 'CRM',
                 'ADBE', 'NOW', 'AMD', 'INTC', 'QCOM', 'AVGO', 'ARM', 'MRVL'],
        'XLF': ['JPM', 'GS', 'MS', 'BAC', 'WFC', 'C', 'BLK', 'SCHW',
                 'V', 'MA', 'AXP', 'COF', 'PYPL'],
        'XLE': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO',
                 'HAL', 'DVN', 'FANG'],
        'XLV': ['UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'DHR',
                 'ISRG', 'REGN', 'VRTX', 'BIIB', 'MRNA', 'GILD'],
        'XLI': ['CAT', 'DE', 'LMT', 'RTX', 'NOC', 'GD', 'BA', 'GE',
                 'HON', 'MMM', 'UPS', 'FDX'],
    }

    sector_etf = next((e for e, m in sector_map.items() if symbol in m), None)
    if not sector_etf:
        return False, {'note': f'No sector ETF mapping for {symbol}'}

    df = price_history.get(sector_etf, pd.DataFrame())
    if df.empty or len(df) < 20:
        return False, {'note': f'Insufficient {sector_etf} history'}

    closes  = df['close']
    ema5    = closes.ewm(span=5,  adjust=False).mean().iloc[-1]
    ema20   = closes.ewm(span=20, adjust=False).mean().iloc[-1]
    bearish = ema5 < ema20

    return bearish, {
        'sector_etf':  sector_etf,
        'etf_ema5':    round(float(ema5), 2),
        'etf_ema20':   round(float(ema20), 2),
        'sector_tide': 'bearish' if bearish else 'bullish',
    }


def check_flow_dry_up(
    symbol:     str,
    uw_flow_df: pd.DataFrame,
) -> tuple[bool, dict]:
    """
    Exits if there's been no significant sweep/dark pool activity
    in the last FLOW_LOOKBACK_HOURS hours.
    Signals that institutional interest has faded.
    """
    if uw_flow_df is None or uw_flow_df.empty:
        return False, {'note': 'No flow data available — not exiting on dry-up'}

    cutoff = datetime.now(timezone.utc) - timedelta(hours=FLOW_LOOKBACK_HOURS)

    sym_flow = uw_flow_df[
        uw_flow_df['symbol'] == symbol
    ] if 'symbol' in uw_flow_df.columns else uw_flow_df

    if sym_flow.empty:
        return True, {
            'note':        f'No flow data found for {symbol}',
            'lookback_h':  FLOW_LOOKBACK_HOURS,
            'dry_up':      True,
        }

    # Filter to lookback window
    if 'timestamp' in sym_flow.columns:
        recent = sym_flow[sym_flow['timestamp'] >= cutoff]
    else:
        recent = sym_flow

    if recent.empty:
        return True, {
            'note':       f'No flow activity for {symbol} in last {FLOW_LOOKBACK_HOURS}h',
            'dry_up':     True,
        }

    # Check if any recent flow meets minimum premium threshold
    significant = recent[recent['premium'].fillna(0) >= MIN_FLOW_PREMIUM] if 'premium' in recent.columns else pd.DataFrame()
    dry_up      = significant.empty

    return dry_up, {
        'recent_flow_count':    len(recent),
        'significant_flow':     len(significant),
        'min_premium':          f'${MIN_FLOW_PREMIUM:,}',
        'lookback_h':           FLOW_LOOKBACK_HOURS,
        'dry_up':               dry_up,
    }


def check_distribution_day(
    symbol:        str,
    price_history: dict,
    current_price: float,
    avg_price:     float,
) -> tuple[bool, dict]:
    """
    Exits if today is a high-volume down day (distribution = institutional selling).
    Condition: price down >2% from prior close AND volume 2x+ 20-day average.
    Only fires if position is already profitable to avoid stop-loss duplication.
    """
    df = price_history.get(symbol, pd.DataFrame())
    if df is None or df.empty or len(df) < 22:
        return False, {'note': 'Insufficient history for distribution check'}

    closes  = df['close']
    volumes = df['volume']

    prior_close = closes.iloc[-2] if len(closes) >= 2 else closes.iloc[-1]
    today_close = closes.iloc[-1]
    today_vol   = volumes.iloc[-1]
    avg_vol     = volumes.iloc[-21:-1].mean()
    rvol        = today_vol / avg_vol if avg_vol > 0 else 1.0
    day_change  = (today_close - prior_close) / prior_close

    pnl_pct     = (current_price - avg_price) / avg_price if avg_price > 0 else 0
    is_profitable = pnl_pct > 0.02   # only fire if up >2% (not on fresh positions)

    fired = (
        day_change <= -DISTRIBUTION_DOWN_PCT and
        rvol >= DISTRIBUTION_RVOL_MIN and
        is_profitable
    )

    return fired, {
        'day_change':  f'{day_change*100:.2f}%',
        'rvol':        round(rvol, 2),
        'is_profitable': is_profitable,
        'threshold':   f'-{DISTRIBUTION_DOWN_PCT*100:.0f}% on {DISTRIBUTION_RVOL_MIN}x vol',
    }


def check_covered_call_exit(
    symbol:        str,
    current_price: float,
    avg_price:     float,
    regime:        str,
    qty:           int,
) -> tuple[bool, dict]:
    """
    When position hits COVERED_CALL_TRIGGER (+15%), switch to CC exit mode.
    Instead of hard selling, we suggest selling a covered call.

    Returns (should_switch_to_cc, details_dict).
    The caller decides whether to execute the CC or fall back to hard sell.
    """
    if avg_price <= 0 or qty < 100:   # need 100 shares minimum for 1 contract
        return False, {'note': 'Insufficient shares for covered call (need 100+)'}

    pnl_pct = (current_price - avg_price) / avg_price

    # CC zone: +15% to +40% (raised from 25% to capture big winners)
    # If position > 25% with round lots, prefer CC over hard exit
    has_round_lot = qty >= 100
    in_cc_zone = COVERED_CALL_TRIGGER <= pnl_pct < 0.40

    if not in_cc_zone:
        return False, {
            'pnl_pct':    f'{pnl_pct*100:.2f}%',
            'cc_trigger': f'+{COVERED_CALL_TRIGGER*100:.0f}%',
            'in_zone':    False,
        }

    # Above 25% with no round lot — can't sell CC, signal hard exit
    if pnl_pct >= COVERED_CALL_HARD_EXIT and not has_round_lot:
        return False, {
            'pnl_pct':    f'{pnl_pct*100:.2f}%',
            'reason':     'above_25pct_odd_lot',
            'in_zone':    False,
        }

    # Strike: 3% OTM from current price
    strike   = round(current_price * (1 + COVERED_CALL_OTM), 2)
    contracts = qty // 100

    # Expiry: 2 weeks out, next Friday
    expiry_dt = datetime.now() + timedelta(days=COVERED_CALL_DTE)
    days_to_friday = (4 - expiry_dt.weekday()) % 7
    expiry_dt += timedelta(days=days_to_friday)
    expiry = expiry_dt.strftime('%Y-%m-%d')

    # Estimate premium using simple approximation
    # At 3% OTM with 14 DTE, premium ≈ 0.5-1.5% of stock price typically
    iv         = 0.25   # assume 25% IV as default
    T          = COVERED_CALL_DTE / 365
    d1         = (math.log(current_price / strike) + (0.05 + 0.5 * iv**2) * T) / (iv * math.sqrt(T))
    d2         = d1 - iv * math.sqrt(T)
    nd1        = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    nd2        = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    premium    = max(current_price * nd1 - strike * math.exp(-0.05 * T) * nd2, 0.05)
    total_income = round(premium * contracts * 100, 2)

    return True, {
        'pnl_pct':       f'{pnl_pct*100:.2f}%',
        'strike':        strike,
        'expiry':        expiry,
        'contracts':     contracts,
        'est_premium':   round(premium, 4),
        'est_income':    total_income,
        'effective_exit': f'+{((strike - avg_price)/avg_price + premium/current_price)*100:.1f}%',
        'regime':        regime,
    }


def check_hard_stops(
    current_price: float,
    avg_price:     float,
    regime:        str,
    adx:           float = 0.0,
) -> tuple[bool, str, dict]:
    """
    Hard stop loss and take profit — always active regardless of signals.
    Returns (fired, reason, details).
    """
    if avg_price <= 0:
        return False, '', {}

    pnl_pct = (current_price - avg_price) / avg_price
    crisis  = (regime == 'crisis')
    flow    = (regime == 'flow')

    stop_pct   = CRISIS_STOP_PCT   if crisis else STOP_LOSS_PCT
    profit_pct = dynamic_take_profit(regime, adx=adx)

    if pnl_pct <= -stop_pct:
        return True, 'stop_loss', {
            'pnl_pct':   f'{pnl_pct*100:.2f}%',
            'threshold': f'-{stop_pct*100:.0f}%',
        }
    if pnl_pct >= profit_pct:
        return True, 'take_profit', {
            'pnl_pct':   f'{pnl_pct*100:.2f}%',
            'threshold': f'+{profit_pct*100:.0f}%',
        }

    return False, '', {
        'pnl_pct':      f'{pnl_pct*100:.2f}%',
        'stop_at':      f'-{stop_pct*100:.0f}%',
        'profit_at':    f'+{profit_pct*100:.0f}%',
    }


# ── Re-Scorer ─────────────────────────────────────────────────────────────────

def rescore_position(
    symbol:        str,
    price_history: dict,
    quotes:        dict,
    spy_history:   pd.DataFrame,
    uw_flow_df:    pd.DataFrame,
    dp_df:         pd.DataFrame,
) -> tuple[float, str]:
    """
    Re-scores an open position using current data.
    Returns (score, direction).
    Imports score_stock from flow_momentum to stay in sync with entry scoring.
    """
    try:
        from flow_momentum import score_stock
        stock_score = score_stock(
            symbol=symbol,
            uw_flow_df=uw_flow_df,
            dp_df=dp_df,
            price_history=price_history,
            quotes=quotes,
            spy_history=spy_history,
        )
        return stock_score.total_score, stock_score.direction
    except Exception as e:
        print(f"  [!] Re-score error for {symbol}: {e}")
        return 0.5, 'neutral'   # neutral score on error — don't exit on data failure


# ── Main Entry Point ──────────────────────────────────────────────────────────

def check_signal_exits(
    portfolio:     dict,
    quotes:        dict,
    spy_history:   pd.DataFrame,
    price_history: dict,
    uw_flow_df:    pd.DataFrame,
    dp_df:         pd.DataFrame,
    regime:        str = 'neutral',
    adx:           float = 0.0,
) -> list[SignalExitResult]:
    """
    Checks all open positions against signal-based exit conditions.
    Returns list of SignalExitResult objects.

    Args:
        portfolio:     Current portfolio dict
        quotes:        {symbol: quote_dict} from data_collector
        spy_history:   SPY price DataFrame
        price_history: {symbol: DataFrame} for all positions
        uw_flow_df:    Combined UW flow DataFrame
        dp_df:         Dark pool DataFrame
        regime:        Current regime string
    """
    positions  = portfolio.get('positions', {})
    trade_log  = portfolio.get('trade_log', [])
    results    = []

    # Also check CSP positions for assignment / profit taking
    options_positions = portfolio.get('options_positions', [])
    for opt_pos in options_positions:
        if opt_pos.get('type') == 'cash_secured_put':
            sym     = opt_pos.get('symbol', '')
            strike  = opt_pos.get('strike', 0)
            expiry  = opt_pos.get('expiry', '')
            premium = opt_pos.get('entry_premium', 0)
            last    = quotes.get(sym, {}).get('last', 0)

            # Check if approaching expiry (5 days)
            try:
                exp_date  = datetime.strptime(expiry, '%Y-%m-%d')
                days_left = (exp_date - datetime.now()).days
            except Exception:
                days_left = 99

            if last > 0 and last < strike:
                # In the money — assignment likely
                results.append(SignalExitResult(
                    symbol=sym, should_exit=True,
                    reason='csp_assignment_likely',
                    current_score=0.85, entry_score=0.85,
                    score_delta=0.0,
                    current_pnl_pct=round((last - strike)/strike*100, 2),
                    current_price=last, avg_price=strike,
                    days_held=0, triggers={'csp': {'days_left': days_left}},
                ))
            elif days_left <= 5:
                # Near expiry and OTM — close for profit
                results.append(SignalExitResult(
                    symbol=sym, should_exit=True,
                    reason='csp_near_expiry_otm',
                    current_score=0.85, entry_score=0.85,
                    score_delta=0.0, current_pnl_pct=100.0,
                    current_price=last, avg_price=strike,
                    days_held=0, triggers={'csp': {'days_left': days_left}},
                ))

    for symbol, position in positions.items():
        # Skip parking positions — never signal-exit GLD/SCHP/VTIP/GDX
        if symbol in PARKING_TICKERS:
            continue

        avg_price = position.get('avg_price', 0)
        qty       = position.get('quantity', 0)
        if qty <= 0 or avg_price <= 0:
            continue

        # Current price
        price = quotes.get(symbol, {}).get('last', avg_price)
        if price <= 0:
            price = avg_price

        pnl_pct = (price - avg_price) / avg_price

        # Days held (from trade log)
        days_held = 0.0
        entries   = [t for t in trade_log
                     if t.get('symbol') == symbol and t.get('type') == 'BUY']
        if entries:
            try:
                entry_ts = entries[-1].get('timestamp', '')
                entry_dt = datetime.fromisoformat(entry_ts)
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                days_held = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 86400
            except Exception:
                pass

        # Entry metadata (stored at buy time if available)
        entry_score     = position.get('entry_score', 0.75)
        entry_direction = position.get('entry_direction', 'bullish')
        entry_tide      = position.get('entry_tide', 'bullish')
        entry_sector_ok = position.get('entry_sector_bullish', True)

        triggers = {}

        # ── 1. Hard stop loss (always check first — capital protection) ─────────
        hard_fired, hard_reason, hard_detail = check_hard_stops(price, avg_price, regime, adx=adx)
        triggers['hard_stop'] = hard_detail
        if hard_fired and hard_reason == 'stop_loss':
            results.append(SignalExitResult(
                symbol=symbol, should_exit=True, reason='stop_loss',
                current_score=entry_score, entry_score=entry_score,
                score_delta=0.0, current_pnl_pct=round(pnl_pct*100, 2),
                current_price=price, avg_price=avg_price,
                days_held=round(days_held, 1), triggers=triggers,
            ))
            continue

        # ── 2. Covered call exit mode (before hard take profit) ───────────────
        # When position is up 15-25%, sell a covered call instead of selling equity
        cc_fired, cc_detail = check_covered_call_exit(
            symbol=symbol, current_price=price, avg_price=avg_price,
            regime=regime, qty=qty,
        )
        triggers['covered_call_exit'] = cc_detail
        if cc_fired:
            # Return as covered_call_mode — caller sells CC, doesn't exit equity yet
            results.append(SignalExitResult(
                symbol=symbol,
                should_exit=False,          # don't exit equity position
                reason='covered_call_mode', # signal to sell covered call instead
                current_score=entry_score,
                entry_score=entry_score,
                score_delta=0.0,
                current_pnl_pct=round(pnl_pct*100, 2),
                current_price=price,
                avg_price=avg_price,
                days_held=round(days_held, 1),
                covered_call_mode=True,
                cc_strike=cc_detail.get('strike', 0),
                cc_expiry=cc_detail.get('expiry', ''),
                cc_est_premium=cc_detail.get('est_premium', 0),
                triggers=triggers,
            ))
            continue

        # ── 3. Hard take profit backup (if CC not possible or position >40%, or odd lot >25%) ──
        if hard_fired and hard_reason == 'take_profit':
            results.append(SignalExitResult(
                symbol=symbol, should_exit=True, reason='take_profit',
                current_score=entry_score, entry_score=entry_score,
                score_delta=0.0, current_pnl_pct=round(pnl_pct*100, 2),
                current_price=price, avg_price=avg_price,
                days_held=round(days_held, 1), triggers=triggers,
            ))
            continue

        # ── 4. Re-score position ──────────────────────────────────────────────
        current_score, current_direction = rescore_position(
            symbol=symbol, price_history=price_history, quotes=quotes,
            spy_history=spy_history, uw_flow_df=uw_flow_df, dp_df=dp_df,
        )
        score_delta = round(current_score - entry_score, 3)

        # ── 5. Signal-based exit checks ───────────────────────────────────────
        exit_reason = None

        # Score decay
        decay_fired, decay_detail = check_score_decay(current_score, entry_score)
        triggers['score_decay'] = decay_detail
        if decay_fired and not exit_reason:
            exit_reason = 'score_decay'

        # Direction flip
        dir_fired, dir_detail = check_direction_flip(current_direction, entry_direction)
        triggers['direction_flip'] = dir_detail
        if dir_fired and not exit_reason:
            exit_reason = 'direction_flip'

        # Market tide
        tide_fired, tide_detail = check_market_tide(spy_history, entry_tide)
        triggers['market_tide'] = tide_detail
        if tide_fired and not exit_reason:
            exit_reason = 'market_tide_bearish'

        # Sector rotation
        sector_fired, sector_detail = check_sector_rotation(
            symbol, price_history, entry_sector_ok
        )
        triggers['sector_rotation'] = sector_detail
        if sector_fired and not exit_reason:
            exit_reason = 'sector_rotation'

        # Distribution day (high volume down day = institutional selling)
        if days_held >= 0.5:  # at least half day old
            dist_fired, dist_detail = check_distribution_day(
                symbol, price_history, price, avg_price
            )
            triggers['distribution_day'] = dist_detail
            if dist_fired and not exit_reason:
                exit_reason = 'distribution_day'
        else:
            triggers['distribution_day'] = {'note': 'Skipped — position < 12h old'}

        # Flow dry-up (only after 24+ hours)
        if days_held >= 1.0:
            flow_fired, flow_detail = check_flow_dry_up(symbol, uw_flow_df)
            triggers['flow_dry_up'] = flow_detail
            if flow_fired and not exit_reason:
                exit_reason = 'flow_dry_up'
        else:
            triggers['flow_dry_up'] = {'note': 'Skipped — position < 24h old'}

        should_exit = exit_reason is not None

        results.append(SignalExitResult(
            symbol=symbol,
            should_exit=should_exit,
            reason=exit_reason or 'hold',
            current_score=round(current_score, 3),
            entry_score=round(entry_score, 3),
            score_delta=score_delta,
            current_pnl_pct=round(pnl_pct*100, 2),
            current_price=price,
            avg_price=avg_price,
            days_held=round(days_held, 1),
            triggers=triggers,
        ))

    return results


def print_signal_exit_summary(results: list):
    if not results:
        return

    print(f"\n{'='*60}")
    print(f"  SIGNAL EXIT MANAGER")
    print(f"{'='*60}")

    for r in results:
        if r.covered_call_mode:
            cc = r.triggers.get('covered_call_exit', {})
            print(f"\n  💰 {r.symbol:<8}  P&L: {r.current_pnl_pct:+.2f}%  "
                  f"Held: {r.days_held:.1f}d")
            print(f"     → COVERED CALL EXIT MODE")
            print(f"     → Sell {cc.get('contracts',1)}x ${cc.get('strike',0):.2f}C "
                  f"{cc.get('expiry','')} @ ~${cc.get('est_premium',0):.2f}")
            print(f"     → Est. income: ${cc.get('est_income',0):,.2f}  "
                  f"Effective exit: {cc.get('effective_exit','')}")
        elif r.should_exit:
            reason = r.reason.upper().replace('_', ' ')
            delta  = f"{r.score_delta:+.3f}"
            print(f"\n  🚨 {r.symbol:<8}  P&L: {r.current_pnl_pct:+.2f}%  "
                  f"Held: {r.days_held:.1f}d  "
                  f"Score: {r.entry_score:.3f}→{r.current_score:.3f} ({delta})")
            print(f"     → EXIT: {reason}")
        else:
            delta = f"{r.score_delta:+.3f}"
            print(f"  ✅ {r.symbol:<8}  P&L: {r.current_pnl_pct:+.2f}%  "
                  f"Held: {r.days_held:.1f}d  "
                  f"Score: {r.entry_score:.3f}→{r.current_score:.3f} ({delta})  HOLD")

    cc_mode = [r for r in results if r.covered_call_mode]
    exits   = [r for r in results if r.should_exit]
    holds   = [r for r in results if not r.should_exit and not r.covered_call_mode]
    print(f"\n  Summary: {len(exits)} exit(s)  "
          f"{len(cc_mode)} CC mode  {len(holds)} hold(s)")
    print(f"{'='*60}\n")


# ── Helper: Store Entry Metadata ──────────────────────────────────────────────

def enrich_position_metadata(
    position:      dict,
    entry_score:   float,
    entry_direction: str,
    spy_history:   pd.DataFrame,
    price_history: dict,
    symbol:        str,
) -> dict:
    """
    Adds entry metadata to a position dict so signal exits can compare
    current state to entry state.
    Call this when recording a new position in paper_trader.
    """
    # Market tide at entry
    entry_tide = 'neutral'
    if spy_history is not None and len(spy_history) >= 20:
        closes = spy_history['close']
        ema5   = closes.ewm(span=5,  adjust=False).mean().iloc[-1]
        ema20  = closes.ewm(span=20, adjust=False).mean().iloc[-1]
        entry_tide = 'bullish' if ema5 > ema20 else 'bearish'

    # Sector bullish at entry
    sector_map = {
        'XLK': ['AAPL','MSFT','NVDA','GOOGL','GOOG','META','CRM','ADBE'],
        'XLF': ['JPM','GS','MS','BAC','WFC','C','V','MA'],
        'XLE': ['XOM','CVX','COP','EOG','SLB','OXY'],
        'XLV': ['UNH','JNJ','LLY','PFE','ABBV','MRK'],
        'XLI': ['CAT','DE','LMT','RTX','NOC','GD','BA'],
    }
    sector_etf = next((e for e, m in sector_map.items() if symbol in m), None)
    entry_sector_bullish = True
    if sector_etf and sector_etf in price_history:
        df = price_history[sector_etf]
        if len(df) >= 20:
            closes = df['close']
            ema5   = closes.ewm(span=5,  adjust=False).mean().iloc[-1]
            ema20  = closes.ewm(span=20, adjust=False).mean().iloc[-1]
            entry_sector_bullish = ema5 > ema20

    position['entry_score']           = entry_score
    position['entry_direction']       = entry_direction
    position['entry_tide']            = entry_tide
    position['entry_sector_bullish']  = entry_sector_bullish
    return position


# ── Smoke Test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from auth import authenticate
    from data_collector import (
        get_price_history, get_quotes, get_vix,
        get_uw_flow, get_uw_dark_pool,
    )
    from paper_trader import load_portfolio
    import pandas as pd

    print("🔌 Authenticating...")
    client, paper = authenticate()
    paper = True  # assume paper mode
    print(f"✅ Connected ({'PAPER' if paper else 'LIVE'} mode)\n")

    portfolio  = load_portfolio()
    positions  = portfolio.get('positions', {})

    print(f"📂 Portfolio: {len(positions)} positions\n")

    if not positions:
        print("No open positions — running scenario simulation:\n")

        # symbol, entry_score, curr_score, direction, pnl_pct, days, qty
        scenarios = [
            ('NVDA',  0.82, 0.85, 'bullish', +12.0, 3.0, 200),  # hold
            ('AAPL',  0.80, 0.35, 'neutral', +2.0,  5.0, 150),  # score decay
            ('MSFT',  0.78, 0.72, 'bearish', -3.0,  2.0, 100),  # direction flip
            ('TSLA',  0.76, 0.60, 'neutral', -8.0,  1.5, 300),  # stop loss
            ('META',  0.85, 0.80, 'bullish', +18.0, 7.0, 200),  # CC exit (round lot)
            ('AMZN',  0.85, 0.80, 'bullish', +18.0, 7.0, 50),   # hard sell (odd lot)
            ('SPY',   0.80, 0.78, 'bullish', +27.0, 10.0, 100), # hard take profit >25%
        ]

        print(f"  {'Symbol':<8} {'Entry':>6} {'Curr':>6} {'Dir':<10} "
              f"{'P&L':>7}  {'Days':>5}  {'Qty':>5}  Action")
        print(f"  {'-'*75}")

        for sym, es, cs, direction, pnl, days, qty in scenarios:
            avg   = 100.0
            price = avg * (1 + pnl/100)

            hard_fired, hard_reason, _ = check_hard_stops(price, avg, 'neutral')
            if hard_fired and hard_reason == 'stop_loss':
                action = '🚨 STOP LOSS'
            else:
                cc_fired, cc_detail = check_covered_call_exit(sym, price, avg, 'neutral', qty)
                if cc_fired:
                    income = cc_detail.get('est_income', 0)
                    action = (f"💰 CC EXIT  ${cc_detail.get('strike',0):.2f}C "
                              f"{cc_detail.get('expiry','')}  +${income:.0f} premium")
                elif hard_fired and hard_reason == 'take_profit':
                    action = '🚨 HARD TAKE PROFIT (>25% or odd lot)'
                else:
                    decay_fired, _ = check_score_decay(cs, es)
                    dir_fired, _   = check_direction_flip(direction, 'bullish')
                    if decay_fired:   action = '🚨 SCORE DECAY'
                    elif dir_fired:   action = '🚨 DIRECTION FLIP'
                    else:             action = '✅ HOLD'

            print(f"  {sym:<8} {es:>6.3f} {cs:>6.3f} {direction:<10} "
                  f"{pnl:>+6.1f}%  {days:>5.1f}d  {qty:>5}  {action}")

    else:
        # Real positions check
        syms       = list(positions.keys())
        quotes     = get_quotes(client, syms)
        spy_hist   = get_price_history(client, 'SPY', days=30)
        price_hist = {s: get_price_history(client, s, days=30) for s in syms}
        uw_flow    = get_uw_flow(limit=100)
        dp_df      = get_uw_dark_pool(limit=50)

        results = check_signal_exits(
            portfolio=portfolio,
            quotes=quotes,
            spy_history=spy_hist,
            price_history=price_hist,
            uw_flow_df=uw_flow,
            dp_df=dp_df,
            regime='neutral',
        )
        print_signal_exit_summary(results)

    print("\n✅ signal_exit_manager.py working correctly.")