"""
flow_momentum.py
Scores every stock in the universe across 9 signals and returns the
top candidates for the current trading cycle.

Qualification rules:
  - Score must be >= MIN_SCORE (0.80) to qualify
  - Maximum MAX_CANDIDATES (4) stocks per cycle
  - Hard blockers from risk_manager must pass before scoring counts

Signal weights (must sum to 1.0):
  1. Market tide (SPY EMA)         0.27
  2. Sector tide                   0.22
  3. Dark pool prints              0.11
  4. Politician flow               0.11
  5. UW sweep / repeated hits      0.10
  6. Corporate insider flow        0.10
  7. Price / RVOL confirmation     0.05
  8. GEX / greek exposure          0.03
  9. ETF inflow / outflow          0.01

Politician boosters (applied to signal 3):
  Committee chair trade   1.5x
  Premium > $50k          1.3x
  Multi-politician        2.0x
  Filed within 5 days     1.4x

PATCH (2026-05-26): three signal scoring fixes after live diagnosis showed
sig_dark_pool, sig_sweep_flow, and sig_insider returning 0 for every ticker:

  1. score_sweep_flow: 4-hour timestamp cutoff was too narrow (rejected all
     overnight, weekend, and pre-market data). Extended to 96 hours to
     cover regular weekend (65h) and 3-day holiday weekend (89h) gaps.
  2. score_dark_pool: the snapshot's market-wide 100-row DataFrame rarely
     contained the symbol being scored. Now fetches per-symbol from
     get_uw_dark_pool(symbol) with a session cache to control API load,
     and extends the timestamp window to 96h for the same reason.
  3. score_insider: the /insider/{symbol}/ticker-flow UW endpoint never
     returned data. Now delegates to data_collector.score_sec_insider()
     which uses SEC EDGAR Form 4 filings (already proven working in
     data_collector). Falls back to 0.0 only if SEC path errors.
"""

import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from dotenv import load_dotenv
import os

# PATCH: persist every scoring cycle for out-of-sample analysis
try:
    from score_logger import log_cycle
except Exception:
    def log_cycle(*args, **kwargs): pass   # graceful no-op if module missing

load_dotenv()

UW_API_KEY  = os.getenv('UW_API_KEY')
UW_BASE_URL = 'https://api.unusualwhales.com/api'

# ── Config ────────────────────────────────────────────────────────────────────

MIN_SCORE           = 0.80   # raised from 0.75 — win rate needs improvement
MAX_CANDIDATES      = 4      # base — overridden dynamically by get_max_candidates()
ADX_THRESHOLD       = 20     # below this = choppy/sideways market, skip entries


def compute_adx(df: 'pd.DataFrame', period: int = 14) -> float:
    """
    Compute Average Directional Index (ADX) from OHLCV DataFrame.
    ADX < 20 = choppy/sideways (no trend) -- skip entries
    ADX 20-25 = weak trend forming
    ADX > 25 = strong trend -- best for momentum
    Returns 0.0 if insufficient data.
    """
    try:
        if df is None or len(df) < period + 5:
            return 0.0
        high  = df['high'].values.astype(float)
        low   = df['low'].values.astype(float)
        close = df['close'].values.astype(float)

        # True Range
        tr = []
        for i in range(1, len(close)):
            tr.append(max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i]  - close[i-1])
            ))

        # Directional Movement
        plus_dm, minus_dm = [], []
        for i in range(1, len(close)):
            up   = high[i] - high[i-1]
            down = low[i-1] - low[i]
            plus_dm.append(up   if up > down and up > 0   else 0.0)
            minus_dm.append(down if down > up and down > 0 else 0.0)

        # Smooth with Wilder's EMA
        def wilder_smooth(vals, n):
            result = [sum(vals[:n])]
            for v in vals[n:]:
                result.append(result[-1] - result[-1] / n + v)
            return result

        atr   = wilder_smooth(tr,       period)
        p_dm  = wilder_smooth(plus_dm,  period)
        m_dm  = wilder_smooth(minus_dm, period)

        # DI+ and DI-
        di_plus  = [100 * p / a if a > 0 else 0 for p, a in zip(p_dm, atr)]
        di_minus = [100 * m / a if a > 0 else 0 for m, a in zip(m_dm, atr)]

        # DX and ADX
        dx = [abs(p - m) / (p + m) * 100 if (p + m) > 0 else 0
              for p, m in zip(di_plus, di_minus)]
        adx = wilder_smooth(dx, period)
        return round(adx[-1], 1) if adx else 0.0
    except Exception:
        return 0.0

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

# PATCH (2026-05-26): scoring window for sweep_flow and dark_pool
# Was: 4h sweep, 8h dark_pool — too narrow, killed all signals in pre-market
# Now: 96h — covers worst-case market closure gaps:
#   - Regular weekend: Fri close → Mon pre-market = 65h
#   - 3-day holiday:   Thu close → Tue pre-market = 89h (e.g. Memorial Day)
#   - 96h provides a safety margin above the 89h worst case.
# Anything older than 4 days is genuinely stale and shouldn't influence
# current scoring decisions.
SWEEP_FLOW_WINDOW_HOURS = 96
DARK_POOL_WINDOW_HOURS  = 96

WEIGHTS = {
    'sweep_flow':    0.10,   # UW options sweep flow
    'dark_pool':     0.11,   # ⬆️ +0.01 from WSB reallocation — strong correlation
    'politician':    0.11,   # ⬆️ +0.01 from WSB reallocation — strongest edge signal
    'insider':       0.10,   # SEC EDGAR Form 4 live
    'price_rvol':    0.05,   # price + relative volume
    'gex':           0.03,   # gamma exposure proxy
    'market_tide':   0.27,   # SPY EMA trend — kept dominant
    'sector_tide':   0.22,   # sector ETF trend — kept dominant
    'etf_flow':      0.01,   # ETF volume flow
    # reddit_wsb REMOVED from scoring weights (edge: +0.022, corr: +0.044 — not significant)
    # WSB now used as top-of-funnel universe injection trigger only
}
# market_tide + sector_tide = 49% — dominant
# politician + insider + dark_pool = 32% — elevated alternative data
# weights sum = 1.0
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


# ── PATCH: per-cycle dark pool cache ──────────────────────────────────────────
# The scoring engine calls score_dark_pool() once per ticker. The old code
# relied on a market-wide DataFrame from snapshot which rarely matched the
# scored symbol. We now fetch per-symbol but cache within the run to avoid
# duplicate API calls if the same symbol is scored twice in one cycle.

_DARK_POOL_CACHE: dict = {}        # symbol -> (timestamp, DataFrame)
_DARK_POOL_CACHE_TTL = 300         # 5 minutes — covers a full cycle's scoring loop


def _get_dark_pool_for_symbol(symbol: str) -> pd.DataFrame:
    """Fetch dark pool for one symbol with short-lived cache."""
    now = time.time()
    cached = _DARK_POOL_CACHE.get(symbol)
    if cached and (now - cached[0]) < _DARK_POOL_CACHE_TTL:
        return cached[1]

    try:
        from data_collector import get_uw_dark_pool
        df = get_uw_dark_pool(symbol=symbol, limit=50)
    except Exception as e:
        print(f"  [!] dark pool fetch error ({symbol}): {e}")
        df = pd.DataFrame()

    _DARK_POOL_CACHE[symbol] = (now, df)
    return df


# ── Signal 1: UW Sweep / Repeated Hits (weight 0.10) ─────────────────────────

def score_sweep_flow(symbol: str, uw_flow_df: pd.DataFrame) -> tuple[float, str]:
    """
    Scores based on options sweep orders and repeated hits.
    Returns (score 0-1, direction).

    PATCH (2026-05-26): timestamp window extended from 4h to 96h. The 4h
    cutoff rejected all overnight, after-hours, and early pre-market data.
    96h covers the 3-day-weekend worst case (Thursday close to Tuesday
    9:26am pre-market = 89h). Regular weekends (65h) and after-hours
    pre-market are also covered.
    """
    if uw_flow_df is None or uw_flow_df.empty:
        return 0.0, 'neutral'

    df = uw_flow_df[uw_flow_df['symbol'] == symbol].copy() if 'symbol' in uw_flow_df.columns else uw_flow_df.copy()
    if df.empty:
        return 0.0, 'neutral'

    # PATCH: 96-hour window (was 4h). Captures prior session across regular weekends AND
    # 3-day holiday weekends (e.g. Memorial Day = 89h gap).
    cutoff = datetime.now(timezone.utc) - timedelta(hours=SWEEP_FLOW_WINDOW_HOURS)
    if 'timestamp' in df.columns:
        # Ensure timestamp is tz-aware UTC for safe comparison
        try:
            ts = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            df = df.assign(timestamp=ts)
            df = df[df['timestamp'].notna() & (df['timestamp'] >= cutoff)]
        except Exception:
            # If anything goes wrong with timestamp parsing, fall through with all rows
            pass

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


# ── Signal 2: Dark Pool Prints (weight 0.11) ──────────────────────────────────

def score_dark_pool(symbol: str, dp_df: pd.DataFrame) -> float:
    """
    Scores based on dark pool block prints.
    Large off-exchange prints signal institutional accumulation.

    PATCH (2026-05-26): the snapshot passes a market-wide 100-row dark pool
    DataFrame. With ~9,000 listed stocks, that 100-row sample rarely
    contains the symbol being scored — every score was 0. This function
    now fetches the symbol's own dark pool data directly (cached within
    the cycle) and uses a 96h window to cover holiday weekend gaps.

    The dp_df argument is still accepted for backward compatibility but
    ignored — kept so score_stock's existing call signature works.
    """
    # PATCH: fetch per-symbol from UW (cached) instead of filtering passed-in DF
    df = _get_dark_pool_for_symbol(symbol)
    if df is None or df.empty:
        return 0.0

    # PATCH: 96-hour window (was 8h) — same weekend-gap reason as sweep_flow
    cutoff = datetime.now(timezone.utc) - timedelta(hours=DARK_POOL_WINDOW_HOURS)
    if 'timestamp' in df.columns:
        try:
            ts = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            df = df.assign(timestamp=ts)
            df = df[df['timestamp'].notna() & (df['timestamp'] >= cutoff)]
        except Exception:
            pass

    if df.empty:
        return 0.0

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


# ── Signal 3: Politician Flow (weight 0.11) ───────────────────────────────────

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
    Scores based on recent insider buying via SEC EDGAR Form 4 filings.

    PATCH (2026-05-26): old implementation called UW /insider/{symbol}/ticker-flow
    which never returned data — every score was 0. Switched to
    data_collector.score_sec_insider() which uses SEC EDGAR Form 4 (free,
    no auth, already proven working).
    """
    try:
        from data_collector import score_sec_insider
        return float(score_sec_insider(symbol, days=14))
    except Exception as e:
        print(f"  [!] score_insider SEC path failed ({symbol}): {e}")
        return 0.0


# ── Signal 5: Price / RVOL Confirmation (weight 0.05) ────────────────────────

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


# ── Signal 6: GEX / Greek Exposure (weight 0.03) ─────────────────────────────

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


# ── Signal 7: Market Tide (weight 0.27) ──────────────────────────────────────

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


# ── Signal 8: Sector Tide (weight 0.22) ──────────────────────────────────────

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


# ── Signal 9: ETF Inflow / Outflow (weight 0.01) ─────────────────────────────

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

    NOTE: dp_df argument is preserved for backward compatibility but is now
    ignored by score_dark_pool() (which fetches per-symbol). main.py and
    pre_market_scanner.py can continue to pass the market-wide dp_df with no
    behavior change.
    """
    signals  = {}
    weighted = {}

    # 1. Sweep flow
    s1, direction = score_sweep_flow(symbol, uw_flow_df)
    signals['sweep_flow']  = s1
    weighted['sweep_flow'] = s1 * WEIGHTS['sweep_flow']

    # 2. Dark pool — dp_df ignored, fetched per-symbol inside the function
    s2 = score_dark_pool(symbol, dp_df)
    signals['dark_pool']  = s2
    weighted['dark_pool'] = s2 * WEIGHTS['dark_pool']

    # 3. Politician
    s3 = score_politician(symbol)
    signals['politician']  = s3
    weighted['politician'] = s3 * WEIGHTS['politician']

    # 4. Insider — now via SEC EDGAR Form 4
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

    # Direction gate: require bullish direction (not bearish)
    # 'neutral' direction means UW data unavailable — don't block entries
    # Only block explicitly bearish signals
    direction_ok = True
    if REQUIRE_DIRECTION and direction == 'bearish':
        direction_ok = False   # explicitly bearish = skip

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
    # PATCH (2026-05-26): clear per-symbol dark pool cache at start of each cycle.
    # Each scoring cycle should start with fresh data.
    _DARK_POOL_CACHE.clear()

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

    result = CycleResult(
        candidates=candidates,
        all_scores=all_scores,
        cycle_time=datetime.now().isoformat(),
        regime=regime,
        signals_fired=signals_fired,
    )

    # PATCH: persist this cycle to scores_live.db for out-of-sample analysis.
    # Pull current VIX from snapshot if available; never let logging break the cycle.
    try:
        vix_val = snapshot.get('vix')
        if vix_val is None:
            # snapshot might use 'vixy' key per the smoke test pattern
            vix_val = snapshot.get('vixy')
        log_cycle(
            result,
            regime=regime,
            vix=vix_val,
            portfolio_value=portfolio_value,
            fg_score=fg_score,
            fg_modifier=fg_mod,
            pc_ratio=pc.get('ratio'),
            pc_modifier=pc_mod,
        )
    except Exception as _e:
        print(f"  [score_logger] WARNING: log call failed: {_e}")

    return result


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
    paper = True  # assume paper mode
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