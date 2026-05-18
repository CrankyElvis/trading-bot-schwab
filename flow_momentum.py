"""
flow_momentum.py
Scores every stock in the universe across 9 signals and returns the
top candidates for the current trading cycle.

Qualification rules:
  - Score must be >= MIN_SCORE (0.75) to qualify
  - Maximum MAX_CANDIDATES (4) stocks per cycle
  - Hard blockers from risk_manager must pass before scoring counts

Signal weights (must sum to 1.0):
  1. UW sweep / repeated hits     0.30
  2. Dark pool prints              0.15
  3. Politician flow               0.15
  4. Corporate insider flow        0.10
  5. Price / RVOL confirmation     0.10
  6. GEX / greek exposure          0.05
  7. Market tide (SPY EMA)         0.05
  8. Sector tide                   0.05
  9. ETF inflow / outflow          0.05

Politician boosters (applied to signal 3):
  Committee chair trade   1.5x
  Premium > $50k          1.3x
  Multi-politician        2.0x
  Filed within 5 days     1.4x
"""

import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from dotenv import load_dotenv
import os

load_dotenv()

UW_API_KEY  = os.getenv('UW_API_KEY')
UW_BASE_URL = 'https://api.unusualwhales.com/api'

# ── Config ────────────────────────────────────────────────────────────────────

MIN_SCORE           = 0.80   # raised from 0.75 — win rate needs improvement
MAX_CANDIDATES      = 4      # base — overridden dynamically by get_max_candidates()

# Dynamic candidate scaling — grows with portfolio
CANDIDATE_TIERS = [
    (200_000, 8),   # > $200k  → 8 candidates
    (100_000, 7),   # > $100k  → 7 candidates
    ( 60_000, 6),   # > $60k   → 6 candidates
    ( 30_000, 5),   # > $30k   → 5 candidates
    (      0, 4),   # default  → 4 candidates
]

def get_max_candidates(portfolio_value: float) -> int:
    """Returns MAX_CANDIDATES scaled to current portfolio size."""
    for threshold, candidates in CANDIDATE_TIERS:
        if portfolio_value >= threshold:
            return candidates
    return 4
MIN_SIGNALS_FIRING  = 3       # at least 3 of 10 signals must score > 0.30
MIN_SWEEP_PREMIUM   = 500_000 # minimum UW sweep premium to count ($500k)
REQUIRE_DIRECTION   = True    # only enter bullish signals in flow/neutral

WEIGHTS = {
    'sweep_flow':    0.10,   # UW options sweep flow
    'dark_pool':     0.10,   # UW dark pool prints
    'politician':    0.10,   # ⬆️ modest increase — real congressional data live
    'insider':       0.10,   # ⬆️ modest increase — real SEC EDGAR Form 4 live
    'price_rvol':    0.05,   # price + relative volume
    'gex':           0.03,   # gamma exposure proxy
    'market_tide':   0.27,   # SPY EMA trend — kept dominant
    'sector_tide':   0.22,   # sector ETF trend — kept dominant
    'etf_flow':      0.01,   # ETF volume flow — reduced to make room
    'reddit_wsb':    0.02,   # Reddit WSB contrarian sentiment
}
# market_tide + sector_tide = 49% — still dominant
# politician + insider = 20% — modest raise from 16%
# weights sum = 1.0
# Note: weights sum to 1.0
# market_tide + sector_tide dominate (51%) — confirmed best predictors
# politician + insider now have real data — weights raised from 0.05 to 0.08

assert abs(sum(WEIGHTS.values()) - 1.0) < 0.001, "Weights must sum to 1.0"

# Sector ETF map — used for sector tide signal
SECTOR_ETFS = {
    'XLK': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META'],
    'XLF': ['JPM', 'GS', 'BAC'],
    'XLE': ['XOM', 'CVX'],
    'XLV': ['JNJ', 'UNH', 'PFE'],
    'XLI': ['CAT', 'DE', 'GE'],
}


# ── Result Container ──────────────────────────────────────────────────────────

@dataclass
class StockScore:
    symbol:      str
    total_score: float
    signals:     dict = field(default_factory=dict)   # {signal_name: raw_score 0-1}
    weighted:    dict = field(default_factory=dict)   # {signal_name: weighted contribution}
    qualifies:   bool = False
    direction:   str  = 'neutral'   # 'bullish' | 'bearish' | 'neutral'
    timestamp:   str  = ''


@dataclass
class CycleResult:
    candidates:     list        # list of StockScore, qualified and sorted
    all_scores:     list        # all scored stocks
    cycle_time:     str
    regime:         str
    signals_fired:  int         # how many stocks had any signal at all


# ── UW Helper ─────────────────────────────────────────────────────────────────

def _uw(endpoint: str, params: dict = None) -> dict:
    try:
        headers = {'Authorization': f'Bearer {UW_API_KEY}'}
        r = requests.get(
            f"{UW_BASE_URL}{endpoint}",
            headers=headers,
            params=params or {},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
        return {}
    except Exception:
        return {}


# ── Signal 1: UW Sweep / Repeated Hits (weight 0.30) ─────────────────────────

def score_sweep_flow(symbol: str, uw_flow_df: pd.DataFrame) -> tuple[float, str]:
    """
    Scores based on options sweep orders and repeated hits.
    Returns (score 0-1, direction).
    """
    if uw_flow_df is None or uw_flow_df.empty:
        return 0.0, 'neutral'

    df = uw_flow_df[uw_flow_df['symbol'] == symbol].copy() if 'symbol' in uw_flow_df.columns else uw_flow_df.copy()
    if df.empty:
        return 0.0, 'neutral'

    # Filter to last 4 hours
    cutoff = datetime.now(timezone.utc) - timedelta(hours=4)
    if 'timestamp' in df.columns:
        df = df[df['timestamp'] >= cutoff]

    if df.empty:
        return 0.0, 'neutral'

    score = 0.0
    call_premium = 0.0
    put_premium  = 0.0

    for _, row in df.iterrows():
        alert  = str(row.get('alert_type', '')).lower()
        prem   = float(row.get('premium', 0) or 0)
        opt    = str(row.get('type', '')).upper()

        # Sweep or repeated hit — filter by minimum premium threshold
        if 'sweep' in alert or 'repeatedhits' in alert or 'repeated' in alert:
            if prem < MIN_SWEEP_PREMIUM:
                continue   # skip small retail flow — not institutional
            if prem >= 2_000_000:
                score += 0.50
            elif prem >= 1_000_000:
                score += 0.40
            elif prem >= 500_000:
                score += 0.25

        # Descending fill (aggressive buyer)
        if 'descending' in alert:
            score += 0.15

        # Track call vs put premium for direction
        if opt == 'CALL':
            call_premium += prem
        elif opt == 'PUT':
            put_premium += prem

    # Direction
    total_prem = call_premium + put_premium
    if total_prem > 0:
        call_ratio = call_premium / total_prem
        if call_ratio >= 0.65:
            direction = 'bullish'
        elif call_ratio <= 0.35:
            direction = 'bearish'
        else:
            direction = 'neutral'
    else:
        direction = 'neutral'

    return min(score, 1.0), direction


# ── Signal 2: Dark Pool Prints (weight 0.15) ──────────────────────────────────

def score_dark_pool(symbol: str, dp_df: pd.DataFrame) -> float:
    """
    Scores based on dark pool block prints.
    Large off-exchange prints signal institutional accumulation.
    """
    if dp_df is None or dp_df.empty:
        return 0.0

    df = dp_df[dp_df['symbol'] == symbol] if 'symbol' in dp_df.columns else dp_df
    if df.empty:
        return 0.0

    cutoff = datetime.now(timezone.utc) - timedelta(hours=8)
    if 'timestamp' in df.columns:
        df = df[df['timestamp'] >= cutoff]

    score = 0.0
    for _, row in df.iterrows():
        prem = float(row.get('premium', 0) or 0)
        if prem >= 5_000_000:
            score += 0.50
        elif prem >= 1_000_000:
            score += 0.30
        elif prem >= 500_000:
            score += 0.15
        elif prem >= 100_000:
            score += 0.05

    return min(score, 1.0)


# ── Signal 3: Politician Flow (weight 0.15) ───────────────────────────────────

def score_politician(symbol: str) -> float:
    """
    Scores based on recent congressional trading activity.
    Applies boosters for committee chairs, large premium, multi-politician, recent filing.
    """
    data  = _uw('/congress/recent-trades')
    items = data.get('data', [])
    if not items:
        return 0.0

    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=30)
    score  = 0.0

    for item in items:
        ticker = (item.get('ticker') or '').upper()
        if ticker != symbol:
            continue

        # Parse filing date
        filed_str = item.get('filed_at') or item.get('transaction_date') or ''
        try:
            filed_dt = datetime.fromisoformat(filed_str.replace('Z', '+00:00'))
            if filed_dt.tzinfo is None:
                filed_dt = filed_dt.replace(tzinfo=timezone.utc)
            if filed_dt < cutoff:
                continue
        except Exception:
            continue

        base  = 0.20
        multi = 1.0

        # Booster: filed within 5 days
        days_since = (now - filed_dt).days
        if days_since <= 5:
            multi *= 1.4

        # Booster: committee chair (check description/tags)
        politician = (item.get('politician') or item.get('representative') or '').lower()
        committees = (item.get('committees') or '').lower()
        if 'chair' in committees or 'chair' in politician:
            multi *= 1.5

        # Booster: premium > $50k
        amount = float(item.get('amount') or item.get('value') or 0)
        if amount >= 50_000:
            multi *= 1.3

        score += base * min(multi, 4.0)   # cap total booster at 4x

    # Multi-politician booster: if 2+ politicians traded same symbol
    politicians_in = [
        i for i in items
        if (i.get('ticker') or '').upper() == symbol
    ]
    if len(politicians_in) >= 2:
        score *= 2.0

    return min(score, 1.0)


# ── Signal 4: Corporate Insider Flow (weight 0.10) ────────────────────────────

def score_insider(symbol: str) -> float:
    """
    Scores based on recent insider buying (not selling — sells are noise).
    """
    data  = _uw(f'/insider/{symbol}/ticker-flow')
    items = data.get('data', [])
    if not items:
        return 0.0

    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=14)
    score  = 0.0

    for item in items:
        action = (item.get('transaction_type') or item.get('type') or '').lower()
        if 'buy' not in action and 'purchase' not in action:
            continue   # ignore sales

        date_str = item.get('filed_at') or item.get('date') or ''
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt < cutoff:
                continue
        except Exception:
            continue

        value = float(item.get('value') or item.get('amount') or 0)
        if value >= 1_000_000:
            score += 0.50
        elif value >= 500_000:
            score += 0.30
        elif value >= 100_000:
            score += 0.15
        elif value >= 50_000:
            score += 0.05

    return min(score, 1.0)


# ── Signal 5: Price / RVOL Confirmation (weight 0.10) ────────────────────────

def score_price_rvol(symbol: str, price_history: dict, quotes: dict) -> float:
    """
    Scores price momentum + relative volume.
    High RVOL + price above 20d EMA = bullish confirmation.
    """
    df = price_history.get(symbol, pd.DataFrame())
    if df.empty or len(df) < 21:
        return 0.0

    score = 0.0
    closes = df['close']

    # 20-day EMA
    ema20 = closes.ewm(span=20, adjust=False).mean().iloc[-1]
    last_close = closes.iloc[-1]
    if last_close > ema20 * 1.02:
        score += 0.40
    elif last_close > ema20:
        score += 0.20

    # 5-day momentum
    if len(closes) >= 6:
        momentum = (last_close - closes.iloc[-6]) / closes.iloc[-6]
        if momentum > 0.03:
            score += 0.30
        elif momentum > 0.01:
            score += 0.15

    # Relative volume
    volumes = df['volume']
    if len(volumes) >= 21:
        avg_vol  = volumes.iloc[-21:-1].mean()
        last_vol = volumes.iloc[-1]
        rvol = last_vol / avg_vol if avg_vol > 0 else 1.0
        if rvol >= 2.0:
            score += 0.30
        elif rvol >= 1.5:
            score += 0.15
        elif rvol >= 1.2:
            score += 0.05

    return min(score, 1.0)


# ── Signal 6: GEX / Greek Exposure (weight 0.05) ─────────────────────────────

def score_gex(symbol: str) -> float:
    """
    Scores based on gamma exposure (GEX).
    Negative GEX = dealers short gamma = amplified moves = good for momentum.
    """
    data = _uw(f'/stock/{symbol}/greek-exposure')
    rows = data.get('data', [])
    if not rows:
        return 0.0

    latest = rows[0] if rows else {}
    try:
        call_gamma = float(latest.get('call_gamma', 0) or 0)
        put_gamma  = float(latest.get('put_gamma',  0) or 0)
        net_gex    = call_gamma + put_gamma   # put_gamma is negative in UW data

        # Negative net GEX = dealers short gamma = more volatile, good for momentum
        if net_gex < -50_000_000:
            return 1.0
        elif net_gex < -10_000_000:
            return 0.60
        elif net_gex < 0:
            return 0.30
        else:
            return 0.10   # positive GEX = pinning, less favorable
    except Exception:
        return 0.0


# ── Signal 7: Market Tide (weight 0.05) ──────────────────────────────────────

def score_market_tide(spy_history: pd.DataFrame) -> float:
    """
    SPY 5d EMA vs 20d EMA.
    Bullish cross = 1.0, bearish = 0.0, neutral = 0.5.
    """
    if spy_history is None or spy_history.empty or len(spy_history) < 20:
        return 0.5

    closes = spy_history['close']
    ema5   = closes.ewm(span=5,  adjust=False).mean().iloc[-1]
    ema20  = closes.ewm(span=20, adjust=False).mean().iloc[-1]

    if ema5 > ema20 * 1.005:
        return 1.0
    elif ema5 > ema20:
        return 0.65
    elif ema5 > ema20 * 0.995:
        return 0.35
    else:
        return 0.0


# ── Signal 8: Sector Tide (weight 0.05) ──────────────────────────────────────

def score_sector_tide(symbol: str, price_history: dict) -> float:
    """
    Checks if the stock's sector ETF is in an uptrend.
    Uses 5d EMA vs 20d EMA on sector ETF.
    """
    # Find which sector ETF covers this symbol
    sector_etf = None
    for etf, members in SECTOR_ETFS.items():
        if symbol in members:
            sector_etf = etf
            break

    if not sector_etf:
        # Default to SPY if sector unknown
        sector_etf = 'SPY'

    df = price_history.get(sector_etf, pd.DataFrame())
    if df.empty or len(df) < 20:
        return 0.5

    closes = df['close']
    ema5   = closes.ewm(span=5,  adjust=False).mean().iloc[-1]
    ema20  = closes.ewm(span=20, adjust=False).mean().iloc[-1]

    if ema5 > ema20 * 1.005:
        return 1.0
    elif ema5 > ema20:
        return 0.65
    elif ema5 > ema20 * 0.995:
        return 0.35
    else:
        return 0.0


# ── Signal 9: ETF Inflow / Outflow (weight 0.05) ─────────────────────────────

def score_etf_flow(symbol: str) -> float:
    """
    Checks ETF inflow/outflow for sector ETFs.
    Positive inflow into relevant sector = tailwind.
    Only applies to ETFs directly; stocks get a pass-through from sector ETF.
    """
    # Determine which ETF to check
    target_etf = None
    if symbol in SECTOR_ETFS:
        target_etf = symbol   # symbol IS an ETF
    else:
        for etf, members in SECTOR_ETFS.items():
            if symbol in members:
                target_etf = etf
                break

    if not target_etf:
        return 0.5   # neutral if no mapping

    data = _uw(f'/etfs/{target_etf}/in-outflow')
    rows = data.get('data', [])
    if not rows:
        return 0.5

    # Use most recent data point
    latest = rows[0] if rows else {}
    try:
        inflow  = float(latest.get('inflow',  0) or 0)
        outflow = float(latest.get('outflow', 0) or 0)
        net     = inflow - outflow

        if net > 500_000_000:
            return 1.0
        elif net > 100_000_000:
            return 0.75
        elif net > 0:
            return 0.55
        elif net > -100_000_000:
            return 0.35
        else:
            return 0.0
    except Exception:
        return 0.5


# ── Score One Stock ───────────────────────────────────────────────────────────

def score_stock(
    symbol:        str,
    uw_flow_df:    pd.DataFrame,
    dp_df:         pd.DataFrame,
    price_history: dict,
    quotes:        dict,
    spy_history:   pd.DataFrame,
) -> StockScore:
    """
    Runs all 9 signals for a single stock and returns a StockScore.
    """
    signals  = {}
    weighted = {}

    # 1. Sweep flow
    s1, direction = score_sweep_flow(symbol, uw_flow_df)
    signals['sweep_flow']  = s1
    weighted['sweep_flow'] = s1 * WEIGHTS['sweep_flow']

    # 2. Dark pool
    s2 = score_dark_pool(symbol, dp_df)
    signals['dark_pool']  = s2
    weighted['dark_pool'] = s2 * WEIGHTS['dark_pool']

    # 3. Politician
    s3 = score_politician(symbol)
    signals['politician']  = s3
    weighted['politician'] = s3 * WEIGHTS['politician']

    # 4. Insider
    s4 = score_insider(symbol)
    signals['insider']  = s4
    weighted['insider'] = s4 * WEIGHTS['insider']

    # 5. Price / RVOL
    s5 = score_price_rvol(symbol, price_history, quotes)
    signals['price_rvol']  = s5
    weighted['price_rvol'] = s5 * WEIGHTS['price_rvol']

    # 6. GEX
    s6 = score_gex(symbol)
    signals['gex']  = s6
    weighted['gex'] = s6 * WEIGHTS['gex']

    # 7. Market tide
    s7 = score_market_tide(spy_history)
    signals['market_tide']  = s7
    weighted['market_tide'] = s7 * WEIGHTS['market_tide']

    # 8. Sector tide
    s8 = score_sector_tide(symbol, price_history)
    signals['sector_tide']  = s8
    weighted['sector_tide'] = s8 * WEIGHTS['sector_tide']

    # 9. ETF flow
    s9 = score_etf_flow(symbol)
    signals['etf_flow']  = s9
    weighted['etf_flow'] = s9 * WEIGHTS['etf_flow']

    total = round(sum(weighted.values()), 4)

    # Confirmation gate: require minimum number of signals firing
    signals_firing = sum(1 for v in signals.values() if v >= 0.30)
    confirmation_ok = signals_firing >= MIN_SIGNALS_FIRING

    # Direction gate: in flow/neutral, require bullish direction
    direction_ok = True
    if REQUIRE_DIRECTION and direction == 'neutral':
        direction_ok = False   # neutral direction = no edge, skip

    qualifies = (total >= MIN_SCORE) and confirmation_ok and direction_ok

    return StockScore(
        symbol=symbol,
        total_score=total,
        signals=signals,
        weighted=weighted,
        qualifies=qualifies,
        direction=direction,
        timestamp=datetime.now().isoformat(),
    )


# ── Score Full Universe ───────────────────────────────────────────────────────

def run_scoring_cycle(
    universe:        list,
    snapshot:        dict,
    regime:          str   = 'neutral',
    portfolio_value: float = 0.0,
) -> CycleResult:
    """
    Scores every symbol in the universe and returns top candidates.
    MAX_CANDIDATES scales dynamically with portfolio value.

    Args:
        universe:        List of ticker symbols to score
        snapshot:        Output of data_collector.collect_snapshot()
        regime:          Current regime string (adjusts urgency/filtering)
        portfolio_value: Current portfolio value for dynamic candidate scaling

    Returns:
        CycleResult with qualified candidates sorted by score descending
    """
    dynamic_max = get_max_candidates(portfolio_value) if portfolio_value > 0 else MAX_CANDIDATES
    uw_flow    = snapshot.get('uw_flow', {})
    dp_df      = snapshot.get('uw_darkpool', pd.DataFrame())
    prices     = snapshot.get('price_history', {})
    quotes     = snapshot.get('quotes', {})
    spy_hist   = prices.get('SPY', pd.DataFrame())

    # Combine all UW flow into one DataFrame for sweep scoring
    flow_frames = [df for df in uw_flow.values() if not df.empty]
    combined_flow = pd.concat(flow_frames, ignore_index=True) if flow_frames else pd.DataFrame()

    # Fear & Greed modifier
    from data_collector import get_fear_greed, fear_greed_modifier, compute_put_call_ratio
    fg       = get_fear_greed()
    fg_score = fg.get('score', 50)
    fg_mod   = fear_greed_modifier(fg_score)
    print(f"  Fear & Greed: {fg_score} ({fg.get('label','Neutral')}) -> score modifier {fg_mod:.2f}x")

    # Put/Call ratio
    pc = compute_put_call_ratio(combined_flow)
    pc_mod = 1.15 if pc['signal'] == 'extreme_fear' else \
             1.05 if pc['signal'] == 'fear' else \
             0.90 if pc['signal'] == 'extreme_greed' else 1.0
    print(f"  Put/Call ratio: {pc['ratio']:.2f} ({pc['signal']}) -> modifier {pc_mod:.2f}x")

    all_scores = []
    signals_fired = 0

    print(f"\n🔍 Scoring {len(universe)} symbols...")
    for symbol in universe:
        s = score_stock(
            symbol=symbol,
            uw_flow_df=combined_flow,
            dp_df=dp_df,
            price_history=prices,
            quotes=quotes,
            spy_history=spy_hist,
        )
        all_scores.append(s)
        # Apply Fear & Greed and P/C modifiers to total score
        s.total_score = round(min(s.total_score * fg_mod * pc_mod, 1.0), 4)
        s.qualifies   = (s.total_score >= MIN_SCORE)

        if s.total_score > 0.1:
            signals_fired += 1
        time.sleep(0.2)   # gentle rate limiting on UW API calls

    # Sort by score descending
    all_scores.sort(key=lambda x: x.total_score, reverse=True)

    # Apply threshold + cap (dynamic based on portfolio size)
    max_n      = dynamic_max if portfolio_value > 0 else MAX_CANDIDATES
    candidates = [s for s in all_scores if s.qualifies][:max_n]

    # In crisis regime, only bullish signals qualify
    if regime == 'crisis':
        candidates = [c for c in candidates if c.direction != 'bearish']

    return CycleResult(
        candidates=candidates,
        all_scores=all_scores,
        cycle_time=datetime.now().isoformat(),
        regime=regime,
        signals_fired=signals_fired,
    )


# ── Display ───────────────────────────────────────────────────────────────────

def print_cycle_result(result: CycleResult):
    print(f"\n{'='*58}")
    print(f"  FLOW MOMENTUM SCORER  —  {result.regime.upper()} REGIME")
    print(f"{'='*58}")
    print(f"  Cycle time:     {result.cycle_time[:19]}")
    print(f"  Stocks scored:  {len(result.all_scores)}")
    print(f"  Signals fired:  {result.signals_fired}")
    print(f"  Threshold:      {MIN_SCORE}  |  Max picks: {MAX_CANDIDATES}")
    print(f"\n  {'Symbol':<8} {'Score':>6}  {'Dir':<10} {'Qualifies'}")
    print(f"  {'-'*40}")
    for s in result.all_scores[:10]:   # show top 10
        qual = '✅ TRADE' if s.qualifies else ''
        print(f"  {s.symbol:<8} {s.total_score:>6.3f}  {s.direction:<10} {qual}")

    if result.candidates:
        print(f"\n  🎯 Qualified candidates ({len(result.candidates)}):")
        for c in result.candidates:
            print(f"\n  {c.symbol}  score={c.total_score:.3f}  direction={c.direction}")
            for sig, val in c.signals.items():
                bar = '█' * int(val * 10)
                print(f"    {sig:<16} {val:.2f}  {bar}")
    else:
        print(f"\n  ⏸  No stocks cleared the {MIN_SCORE} threshold this cycle.")
        print(f"     Cash stays parked in inflation hedge positions.")
    print(f"{'='*58}\n")


# ── Smoke Test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from auth import authenticate
    from data_collector import collect_snapshot, DEFAULT_UNIVERSE, get_vix, get_vix_history
    from regime_engine import evaluate_regime

    print("🔌 Authenticating...")
    client, paper = authenticate()
    print(f"✅ Connected ({'PAPER' if paper else 'LIVE'} mode)\n")

    # Get regime
    from data_collector import get_vix, get_vix_history
    vixy     = get_vix(client)
    vix_hist = get_vix_history(client, days=30)
    regime   = evaluate_regime(vixy, vix_hist)
    print(f"📊 Regime: {regime.regime.upper()}  (VIXY {vixy:.2f})\n")

    # Collect snapshot (use a small universe for smoke test)
    test_universe = ['SPY', 'QQQ', 'AAPL', 'NVDA', 'MSFT', 'JPM', 'GS', 'XLK', 'XLF']
    print(f"📡 Collecting snapshot for {len(test_universe)} symbols...")
    snapshot = collect_snapshot(client, test_universe)

    # Run scoring cycle
    result = run_scoring_cycle(
        universe=test_universe,
        snapshot=snapshot,
        regime=regime.regime,
    )
    print_cycle_result(result)