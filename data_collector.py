"""
data_collector.py
Fetches all market data needed by the trading bot:
  - Price history (OHLCV) from Schwab
  - Real-time quotes from Schwab
  - VIX proxy via VIXY ETF (Schwab does not expose $VIX.X index directly)
  - Options flow (sweeps, dark pool, repeated hits) from Unusual Whales
  - Fear & Greed Index (free, no key required)
  - Put/Call ratio from UW flow data
  - RSI and MACD from Alpha Vantage (free tier)
  - Earnings calendar from Finnhub (free tier)
"""

import os
import time
import json
import sqlite3
from typing import Optional

def _retry(fn, *args, attempts=3, **kwargs):
    """Retry a Schwab API call with exponential backoff on failure."""
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if i < attempts - 1:
                time.sleep(0.4 * (i + 1))
            else:
                raise

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from auth import authenticate

load_dotenv()

UW_API_KEY         = os.getenv('UW_API_KEY')
UW_BASE_URL        = 'https://api.unusualwhales.com/api'
AV_BASE            = 'https://www.alphavantage.co/query'
FH_BASE            = 'https://finnhub.io/api/v1'

# ── SEC EDGAR Configuration ───────────────────────────────────────────────────
# SEC requires a real contact email in the User-Agent header per their fair-access
# policy. Placeholder emails get silently rate-limited or rejected.
# Format: "<identifier> <real-email>". SEC will email this address before blocking
# the IP if your traffic causes problems. Hard rate limit: 10 req/sec per IP.
SEC_USER_AGENT = 'CrankyElvis Trading Bot ben.bissette@gmail.com'
SEC_HEADERS    = {'User-Agent': SEC_USER_AGENT, 'Accept-Encoding': 'gzip, deflate'}
SEC_RATE_DELAY = 0.12   # 0.12s = ~8 req/sec, safely under SEC's 10/sec cap

# ── Universe ──────────────────────────────────────────────────────────────────
# DEFAULT_UNIVERSE used for intraday snapshot (parking tickers + core ETFs)
# Full 150-symbol scan is handled by pre_market_scanner.py at 6am ET
DEFAULT_UNIVERSE = [
    'SPY', 'QQQ', 'IWM', 'DIA',
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI',
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL',
    'META', 'TSLA', 'JPM', 'GS', 'BAC',
    # Parking tickers always included for cash manager
    'GLD', 'GDX', 'SCHP', 'VTIP',
]

VIX_SYMBOL = 'VIXY'   # Schwab doesn't expose $VIX.X; VIXY is the closest ETF proxy


# ── Schwab: Price History ─────────────────────────────────────────────────────

def get_price_history(client, symbol: str, days: int = 60) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: datetime, open, high, low, close, volume
    Pulls daily OHLCV for the last `days` calendar days.
    """
    try:
        end   = datetime.now()
        start = end - timedelta(days=days)

        resp = _retry(client.get_price_history,
            symbol,
            period_type=client.PriceHistory.PeriodType.MONTH,
            period=client.PriceHistory.Period.TWO_MONTHS,
            frequency_type=client.PriceHistory.FrequencyType.DAILY,
            frequency=client.PriceHistory.Frequency.DAILY,
            start_datetime=start,
            end_datetime=end,
        )

        data    = resp.json()
        candles = data.get('candles', [])
        if not candles:
            print(f"  [!] No price history returned for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        return df.sort_values('datetime').reset_index(drop=True)

    except Exception as e:
        print(f"  [!] Price history error for {symbol}: {e}")
        return pd.DataFrame()


# ── Schwab: Real-Time Quote ───────────────────────────────────────────────────

def get_quote(client, symbol: str) -> dict:
    """
    Returns a dict with: symbol, last, bid, ask, volume, mark.
    Falls back through regular -> quote -> extended for after-hours.
    """
    try:
        resp  = _retry(client.get_quote, symbol)
        data  = resp.json()
        asset = data.get(symbol, {})

        candidates = [
            asset.get('regular',  {}).get('regularMarketLastPrice'),
            asset.get('quote',    {}).get('lastPrice'),
            asset.get('quote',    {}).get('mark'),
            asset.get('extended', {}).get('lastPrice'),
        ]
        last = next((float(v) for v in candidates if v and float(v) > 0), 0)

        q   = asset.get('quote', {})
        bid = q.get('bidPrice', 0)
        ask = q.get('askPrice', 0)
        mid = round((bid + ask) / 2, 4) if bid and ask else last

        return {
            'symbol': symbol,
            'last':   last,
            'bid':    bid,
            'ask':    ask,
            'volume': q.get('totalVolume', 0),
            'mark':   q.get('mark', 0),
            'mid':    mid,
        }
    except Exception as e:
        print(f"  [!] Quote error for {symbol}: {e}")
        return {}


def get_quotes(client, symbols: list) -> dict:
    """Batch quote fetch with rate limiting. Returns {symbol: quote_dict}."""
    results = {}
    for symbol in symbols:
        for attempt in range(3):   # retry up to 3x on transient errors
            try:
                results[symbol] = get_quote(client, symbol)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))   # back off: 0.5s, 1.0s
                else:
                    results[symbol] = {}
        time.sleep(0.25)   # 4 req/sec max — well under Schwab's 120/min limit
    return results


# ── Schwab: VIX (VIXY proxy) ──────────────────────────────────────────────────

def get_vix(client) -> float:
    """
    Returns the current VIXY price as a float.
    Falls back through regular -> quote -> extended to handle after-hours.
    """
    try:
        resp  = client.get_quote(VIX_SYMBOL)
        data  = resp.json()
        asset = data.get(VIX_SYMBOL, {})

        candidates = [
            asset.get('regular',  {}).get('regularMarketLastPrice'),
            asset.get('quote',    {}).get('lastPrice'),
            asset.get('quote',    {}).get('mark'),
            asset.get('extended', {}).get('lastPrice'),
        ]
        for val in candidates:
            if val and float(val) > 0:
                return round(float(val), 4)
        return 0.0
    except Exception as e:
        print(f"  [!] VIX fetch error: {e}")
        return 0.0


def get_vix_history(client, days: int = 30) -> pd.DataFrame:
    """
    Returns daily VIXY closes for the last `days` days.
    Used by regime_engine to compute the 30-day rolling average.
    """
    try:
        end   = datetime.now()
        start = end - timedelta(days=days + 10)

        resp = _retry(client.get_price_history,
            VIX_SYMBOL,
            period_type=client.PriceHistory.PeriodType.MONTH,
            period=client.PriceHistory.Period.ONE_MONTH,
            frequency_type=client.PriceHistory.FrequencyType.DAILY,
            frequency=client.PriceHistory.Frequency.DAILY,
            start_datetime=start,
            end_datetime=end,
        )

        data    = resp.json()
        candles = data.get('candles', [])
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df = df[['datetime', 'close']].rename(columns={'close': 'vix'})
        df = df.sort_values('datetime').reset_index(drop=True)
        return df.tail(days)

    except Exception as e:
        print(f"  [!] VIX history error: {e}")
        return pd.DataFrame()


# ── Unusual Whales: Options Flow ──────────────────────────────────────────────

def _uw_headers() -> dict:
    return {
        'Authorization': f'Bearer {UW_API_KEY}',
        'Content-Type':  'application/json',
    }


def get_uw_flow(symbol: str = None, limit: int = 100) -> pd.DataFrame:
    """
    Fetches recent options flow from Unusual Whales.
    If symbol is None, returns market-wide flow.
    """
    try:
        url    = f"{UW_BASE_URL}/stock/{symbol}/flow-alerts" if symbol else f"{UW_BASE_URL}/option-trades/flow-alerts"
        params = {'limit': limit}
        resp   = requests.get(url, headers=_uw_headers(), params=params, timeout=10)
        resp.raise_for_status()

        data = resp.json().get('data', [])
        if not data:
            return pd.DataFrame()

        rows = []
        for item in data:
            rows.append({
                'symbol':        item.get('ticker', symbol),
                'strike':        item.get('strike'),
                'expiry':        item.get('expiry_date') or item.get('expiration_date'),
                'type':          item.get('option_type', '').upper(),
                'side':          item.get('side', '').lower(),
                'premium':       float(item.get('total_premium') or item.get('premium', 0)),
                'size':          int(item.get('size', 0)),
                'volume':        int(item.get('volume', 0)),
                'open_interest': int(item.get('open_interest', 0)),
                'vol_oi_ratio':  float(item.get('volume_oi_ratio', 0)),
                'alert_type':    item.get('alert_rule', '') or item.get('tags', ''),
                'sentiment':     item.get('sentiment', ''),
                'timestamp':     item.get('created_at') or item.get('timestamp', ''),
            })

        df = pd.DataFrame(rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        return df

    except Exception as e:
        print(f"  [!] UW flow error ({symbol or 'market-wide'}): {e}")
        return pd.DataFrame()


def get_uw_dark_pool(symbol: str = None, limit: int = 50) -> pd.DataFrame:
    """Fetches dark pool (off-exchange block) prints from Unusual Whales."""
    try:
        url    = f"{UW_BASE_URL}/stock/{symbol}/darkpool" if symbol else f"{UW_BASE_URL}/darkpool/recent"
        params = {'limit': limit}
        resp   = requests.get(url, headers=_uw_headers(), params=params, timeout=10)
        resp.raise_for_status()

        data = resp.json().get('data', [])
        if not data:
            return pd.DataFrame()

        rows = []
        for item in data:
            rows.append({
                'symbol':    item.get('ticker', symbol),
                'price':     float(item.get('price', 0)),
                'size':      int(item.get('size', 0)),
                'premium':   float(item.get('premium') or item.get('notional_value', 0)),
                'sentiment': item.get('sentiment', ''),
                'timestamp': item.get('executed_at') or item.get('timestamp', ''),
            })

        df = pd.DataFrame(rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        return df

    except Exception as e:
        print(f"  [!] UW dark pool error ({symbol or 'market-wide'}): {e}")
        return pd.DataFrame()


def get_uw_flow_for_universe(symbols: list = None, limit_per: int = 50) -> dict:
    """Pulls options flow for every symbol in the universe. Returns {symbol: DataFrame}."""
    if symbols is None:
        symbols = DEFAULT_UNIVERSE
    results = {}
    for symbol in symbols:
        print(f"  Fetching UW flow: {symbol}")
        results[symbol] = get_uw_flow(symbol, limit=limit_per)
        time.sleep(0.3)
    return results


# ── Relative Volume ───────────────────────────────────────────────────────────

def compute_rvol(price_df: pd.DataFrame, window: int = 20) -> float:
    """Relative volume = today's volume / 20-day average volume."""
    if price_df.empty or len(price_df) < window + 1:
        return 1.0
    avg_vol   = price_df['volume'].iloc[-(window + 1):-1].mean()
    today_vol = price_df['volume'].iloc[-1]
    if avg_vol == 0:
        return 1.0
    return round(today_vol / avg_vol, 2)


# ── Fear & Greed Index (free, no key required) ────────────────────────────────

def get_fear_greed() -> dict:
    """
    Fetches the CNN Fear & Greed Index from feargreedchart.com.
    Completely free, no API key required.
    Returns dict with score (0-100), label, and components.
    Extreme Fear (<20) = buy signal. Extreme Greed (>80) = caution.
    """
    try:
        r = requests.get('https://feargreedchart.com/api/?action=all', timeout=8)
        if r.status_code != 200:
            return {'score': 50, 'label': 'Neutral', 'components': []}
        data  = r.json()
        score = data.get('score', {})
        return {
            'score':      score.get('score', 50),
            'label':      score.get('label', 'Neutral'),
            'components': score.get('components', []),
        }
    except Exception as e:
        print(f"  [!] Fear & Greed fetch error: {e}")
        return {'score': 50, 'label': 'Neutral', 'components': []}


def fear_greed_modifier(score: int) -> float:
    """
    Converts Fear & Greed score to a signal multiplier for flow_momentum.
    Extreme Fear (<20)  = 1.30x  (contrarian buy boost)
    Fear (20-40)        = 1.10x
    Neutral (40-60)     = 1.00x
    Greed (60-80)       = 0.90x
    Extreme Greed (>80) = 0.75x  (tighten threshold)
    """
    if score < 20:   return 1.30
    elif score < 40: return 1.10
    elif score < 60: return 1.00
    elif score < 80: return 0.90
    else:            return 0.75


# ── Put/Call Ratio ────────────────────────────────────────────────────────────

def compute_put_call_ratio(uw_flow_df) -> dict:
    """
    Computes aggregate put/call premium ratio from UW flow data.
    P/C > 1.2 = extreme fear = contrarian buy signal.
    P/C < 0.7 = extreme greed = caution.
    """
    if uw_flow_df is None or uw_flow_df.empty:
        return {'ratio': 1.0, 'signal': 'neutral', 'calls': 0, 'puts': 0,
                'call_premium': 0, 'put_premium': 0}

    df = uw_flow_df.copy()
    if 'type' not in df.columns:
        return {'ratio': 1.0, 'signal': 'neutral', 'calls': 0, 'puts': 0,
                'call_premium': 0, 'put_premium': 0}

    call_prem = df[df['type'].str.upper() == 'CALL']['premium'].sum()
    put_prem  = df[df['type'].str.upper() == 'PUT']['premium'].sum()
    calls     = len(df[df['type'].str.upper() == 'CALL'])
    puts      = len(df[df['type'].str.upper() == 'PUT'])
    ratio     = round(put_prem / call_prem, 3) if call_prem > 0 else 1.0

    if ratio > 1.2:   signal = 'extreme_fear'
    elif ratio > 1.0: signal = 'fear'
    elif ratio > 0.8: signal = 'neutral'
    elif ratio > 0.7: signal = 'greed'
    else:             signal = 'extreme_greed'

    return {
        'ratio':         ratio,
        'signal':        signal,
        'calls':         calls,
        'puts':          puts,
        'call_premium':  round(call_prem, 2),
        'put_premium':   round(put_prem, 2),
    }


# ── Alpha Vantage: Technical Indicators ──────────────────────────────────────

def get_av_rsi(symbol: str, period: int = 14) -> float:
    """
    Fetches RSI from Alpha Vantage free tier.
    Returns latest RSI value (0-100). Returns 50.0 on error.
    RSI < 30 = oversold (buy signal). RSI > 70 = overbought (caution).
    """
    try:
        params = {
            'function':    'RSI',
            'symbol':      symbol,
            'interval':    'daily',
            'time_period': period,
            'series_type': 'close',
            'apikey':      os.getenv('ALPHA_VANTAGE_API_KEY'),
        }
        r      = requests.get(AV_BASE, params=params, timeout=12)
        series = r.json().get('Technical Analysis: RSI', {})
        if not series:
            return 50.0
        latest = sorted(series.keys())[-1]
        return round(float(series[latest]['RSI']), 2)
    except Exception as e:
        print(f"  [!] AV RSI error for {symbol}: {e}")
        return 50.0


def get_av_macd(symbol: str) -> dict:
    """
    Fetches MACD from Alpha Vantage free tier.
    Returns dict with macd, signal, histogram.
    Positive histogram = bullish momentum. Negative = bearish.
    """
    try:
        params = {
            'function':    'MACD',
            'symbol':      symbol,
            'interval':    'daily',
            'series_type': 'close',
            'apikey':      os.getenv('ALPHA_VANTAGE_API_KEY'),
        }
        r      = requests.get(AV_BASE, params=params, timeout=12)
        series = r.json().get('Technical Analysis: MACD', {})
        if not series:
            return {}
        latest = series[sorted(series.keys())[-1]]
        return {
            'macd':      round(float(latest['MACD']), 4),
            'signal':    round(float(latest['MACD_Signal']), 4),
            'histogram': round(float(latest['MACD_Hist']), 4),
        }
    except Exception as e:
        print(f"  [!] AV MACD error for {symbol}: {e}")
        return {}


# ── Finnhub: Earnings Calendar ────────────────────────────────────────────────

def get_finnhub_earnings(symbol: str, days_ahead: int = 7) -> list:
    """
    Fetches upcoming earnings dates from Finnhub free tier.
    Returns list of earnings events within days_ahead window.
    """
    try:
        today  = datetime.now().strftime('%Y-%m-%d')
        end    = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        params = {
            'symbol': symbol,
            'from':   today,
            'to':     end,
            'token':  os.getenv('FINNHUB_API_KEY'),
        }
        r = requests.get(f"{FH_BASE}/calendar/earnings", params=params, timeout=8)
        return r.json().get('earningsCalendar', []) if r.status_code == 200 else []
    except Exception as e:
        print(f"  [!] Finnhub earnings error for {symbol}: {e}")
        return []


def has_earnings_soon(symbol: str, hours: int = 48) -> tuple:
    """
    Returns (True, date_str) if earnings within hours window, else (False, '').
    Used by risk_manager earnings blocker.
    """
    events = get_finnhub_earnings(symbol, days_ahead=max(hours // 24 + 1, 3))
    now    = datetime.now(timezone.utc)
    window = now + timedelta(hours=hours)

    for event in events:
        date_str = event.get('date', '')
        if not date_str:
            continue
        try:
            dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
            if now <= dt <= window:
                return True, date_str
        except Exception:
            continue

    return False, ''


# ── Full Snapshot ─────────────────────────────────────────────────────────────

def collect_snapshot(client, symbols: list = None) -> dict:
    """
    Collects a full data snapshot for the bot's decision cycle.
    Returns all data needed by regime_engine, cash_manager,
    risk_manager, flow_momentum, and exit_manager.
    """
    if symbols is None:
        symbols = DEFAULT_UNIVERSE

    print("\n📡 Collecting market snapshot...")

    print("  Fetching VIX...")
    vixy        = get_vix(client)
    vix_history = get_vix_history(client, days=30)
    print(f"  VIX: {vixy:.2f}")

    print("  Fetching quotes...")
    quotes = get_quotes(client, symbols)

    print("  Fetching price history...")
    price_history = {}
    for sym in symbols:
        for attempt in range(3):
            try:
                price_history[sym] = get_price_history(client, sym, days=60)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                else:
                    price_history[sym] = pd.DataFrame()
        time.sleep(0.3)   # ~3 req/sec for price history (heavier endpoint)

    print("  Fetching UW options flow...")
    uw_flow = get_uw_flow_for_universe(symbols)

    print("  Fetching UW dark pool prints...")
    uw_darkpool = get_uw_dark_pool(limit=100)

    print("  Fetching Fear & Greed Index...")
    fear_greed = get_fear_greed()
    print(f"  Fear & Greed: {fear_greed['score']} ({fear_greed['label']})")

    print("✅ Snapshot complete.\n")

    return {
        'vixy':          vixy,
        'vix_history':   vix_history,
        'quotes':        quotes,
        'price_history': price_history,
        'uw_flow':       uw_flow,
        'uw_darkpool':   uw_darkpool,
        'fear_greed':    fear_greed,
        'timestamp':     datetime.now(),
    }



# ── CBOE VIX Term Structure ───────────────────────────────────────────────────

def get_vix_term_structure() -> dict:
    """
    Fetches VIX9D, VIX (30d), VIX3M from CBOE free CSVs.
    Returns dict with current levels and contango/backwardation signal.

    Contango  (VIX9D < VIX < VIX3M) = calm market, safe to enter
    Backwardation (VIX9D > VIX3M)   = fear spiking, halt new entries
    Inversion = regime change warning

    Signal values:
      'contango'      — normal, entries ok
      'flat'          — neutral
      'backwardation' — fear spike, halt entries
      'inversion'     — extreme warning
    """
    urls = {
        'vix9d': 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv',
        'vix':   'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv',
        'vix3m': 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv',
    }
    levels = {}
    try:
        import io
        for name, url in urls.items():
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.strip().upper() for c in df.columns]
            df['DATE'] = pd.to_datetime(df['DATE'])
            df = df.sort_values('DATE')
            levels[name] = round(float(df['CLOSE'].iloc[-1]), 2)
            time.sleep(0.2)
    except Exception as e:
        print(f"  [!] VIX term structure error: {e}")
        return {'vix9d': 0, 'vix': 0, 'vix3m': 0, 'signal': 'flat', 'spread': 0}

    vix9d = levels.get('vix9d', 0)
    vix30 = levels.get('vix', 0)
    vix3m = levels.get('vix3m', 0)

    if vix9d <= 0 or vix3m <= 0:
        return {'vix9d': vix9d, 'vix': vix30, 'vix3m': vix3m,
                'signal': 'flat', 'spread': 0}

    spread = round(vix9d - vix3m, 2)

    if vix9d > vix3m * 1.10:
        signal = 'backwardation'   # fear spike — halt entries
    elif vix9d > vix3m:
        signal = 'inversion'       # mild warning
    elif vix9d < vix3m * 0.90:
        signal = 'contango'        # calm — safe to enter
    else:
        signal = 'flat'

    return {
        'vix9d':  vix9d,
        'vix':    vix30,
        'vix3m':  vix3m,
        'spread': spread,
        'signal': signal,
        'halt_entries': signal in ('backwardation', 'inversion'),
    }


# ── SEC EDGAR Form 4 (Insider Buying) ─────────────────────────────────────────
# Implementation notes (2026-05-27):
#   Earlier versions hit the full-text search endpoint at efts.sec.gov, which
#   returned 0 hits because Form 4 filings don't contain the ticker symbol in
#   document text — they reference companies by name/CIK. Switched to the
#   submissions API + per-filing XML parsing for ground-truth transaction
#   codes.
#
# Pipeline per ticker:
#   1. Look up CIK from cached ticker→CIK map (refreshed weekly)
#   2. Fetch data.sec.gov/submissions/CIK{cik}.json (per-day cached)
#   3. Filter recent filings to Form 4 within lookback window
#   4. For each Form 4, fetch primary_doc.xml (cached forever by accession)
#   5. Parse transaction codes: P=buy, S=sell. Everything else (A, M, F, G, C)
#      is ignored — those are grants/exercises/tax-withholding, not real-money
#      trades.
#
# Rate budget: SEC allows 10 req/sec. We sleep SEC_RATE_DELAY=0.12s between
# every SEC call. A 42-symbol cycle with ~5 filings/ticker = ~250 calls = ~30s
# worst case, but the accession-level XML cache means after first cycle of the
# day, subsequent cycles only pay for submissions lookups (~6s) and any new
# filings since.

SEC_TICKER_CIK_CACHE   = '/root/trading-bot/data/sec_ticker_cik.json'
SEC_TICKER_CIK_TTL_SEC = 7 * 24 * 3600   # refresh weekly
SEC_FILINGS_CACHE_DB   = '/root/trading-bot/data/sec_filings_cache.db'
SEC_SUBMISSIONS_TTL    = 4 * 3600         # 4 hours — same-day cycles reuse
# Transaction codes that count as real-money trades. P=Purchase, S=Sale.
# A=Award/grant (RSU vesting, option grants — compensation, not a buy).
# M=Exercise (option → stock conversion, not new commitment).
# F=Tax withholding on vest. G=Gift. C=Conversion. I/J=Other.
SEC_BUY_CODES  = {'P'}
SEC_SELL_CODES = {'S'}

# In-memory cache of ticker→CIK loaded from disk on first use
_SEC_TICKER_CIK_MAP: Optional[dict] = None


def _sec_init_cache_db():
    """Create the per-accession XML transaction cache. Idempotent."""
    import os
    os.makedirs(os.path.dirname(SEC_FILINGS_CACHE_DB), exist_ok=True)
    conn = sqlite3.connect(SEC_FILINGS_CACHE_DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    # accession_number is unique per filing; once we parse it, never re-parse.
    conn.execute('''
        CREATE TABLE IF NOT EXISTS form4_transactions (
            accession_number TEXT PRIMARY KEY,
            cik              TEXT,
            symbol           TEXT,
            filing_date      TEXT,
            buy_count        INTEGER,
            sell_count       INTEGER,
            transactions_json TEXT,
            parsed_at        TEXT NOT NULL
        )
    ''')
    # submissions response cache per ticker — short TTL since new filings appear
    conn.execute('''
        CREATE TABLE IF NOT EXISTS submissions_cache (
            symbol      TEXT PRIMARY KEY,
            cik         TEXT,
            fetched_at  TEXT NOT NULL,
            response_json TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def _sec_load_ticker_cik_map() -> dict:
    """
    Returns the ticker→CIK mapping. Lazy-loaded from disk cache, refetched
    from SEC if file is missing or older than SEC_TICKER_CIK_TTL_SEC.
    """
    import os
    global _SEC_TICKER_CIK_MAP
    if _SEC_TICKER_CIK_MAP is not None:
        return _SEC_TICKER_CIK_MAP

    # Check disk cache freshness
    needs_refresh = True
    if os.path.exists(SEC_TICKER_CIK_CACHE):
        age = time.time() - os.path.getmtime(SEC_TICKER_CIK_CACHE)
        if age < SEC_TICKER_CIK_TTL_SEC:
            needs_refresh = False

    if needs_refresh:
        try:
            print('  [SEC] Refreshing ticker→CIK map from SEC...')
            os.makedirs(os.path.dirname(SEC_TICKER_CIK_CACHE), exist_ok=True)
            r = requests.get(
                'https://www.sec.gov/files/company_tickers.json',
                headers=SEC_HEADERS, timeout=20,
            )
            time.sleep(SEC_RATE_DELAY)
            if r.status_code == 200:
                with open(SEC_TICKER_CIK_CACHE, 'w') as f:
                    f.write(r.text)
            else:
                print(f'  [SEC] ticker map fetch failed: HTTP {r.status_code}')
        except Exception as e:
            print(f'  [SEC] ticker map fetch error: {e}')

    # Load from disk and convert to {TICKER_UPPER: cik_padded_10}
    mapping = {}
    try:
        with open(SEC_TICKER_CIK_CACHE) as f:
            raw = json.load(f)
        # SEC format: {"0": {"cik_str": int, "ticker": str, "title": str}, ...}
        for entry in raw.values():
            ticker = str(entry.get('ticker', '')).upper().strip()
            cik    = entry.get('cik_str')
            if ticker and cik is not None:
                mapping[ticker] = f'{int(cik):010d}'
    except Exception as e:
        print(f'  [SEC] ticker map load failed: {e}')

    _SEC_TICKER_CIK_MAP = mapping
    if mapping:
        print(f'  [SEC] Loaded {len(mapping)} ticker→CIK mappings')
    return mapping


def _sec_lookup_cik(symbol: str) -> Optional[str]:
    """Returns 10-digit zero-padded CIK string or None if ticker unknown."""
    mapping = _sec_load_ticker_cik_map()
    return mapping.get(symbol.upper())


def _sec_fetch_submissions(symbol: str, cik: str) -> Optional[dict]:
    """
    Fetch the submissions JSON for a CIK with short-TTL caching.
    Returns parsed dict or None on failure.
    """
    _sec_init_cache_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(SEC_FILINGS_CACHE_DB, timeout=10)
    try:
        row = conn.execute(
            'SELECT fetched_at, response_json FROM submissions_cache WHERE symbol = ?',
            (symbol.upper(),)
        ).fetchone()
        if row:
            try:
                age = (datetime.now(timezone.utc) -
                       datetime.fromisoformat(row[0])).total_seconds()
                if age < SEC_SUBMISSIONS_TTL:
                    return json.loads(row[1])
            except Exception:
                pass
    finally:
        conn.close()

    # Cache miss or stale — fetch fresh
    url = f'https://data.sec.gov/submissions/CIK{cik}.json'
    try:
        r = requests.get(url, headers=SEC_HEADERS, timeout=15)
        time.sleep(SEC_RATE_DELAY)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception as e:
        print(f'  [SEC] submissions fetch failed for {symbol} (CIK {cik}): {e}')
        return None

    conn = sqlite3.connect(SEC_FILINGS_CACHE_DB, timeout=10)
    try:
        conn.execute('''
            INSERT OR REPLACE INTO submissions_cache
            (symbol, cik, fetched_at, response_json)
            VALUES (?, ?, ?, ?)
        ''', (symbol.upper(), cik, now_iso, json.dumps(data)))
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()
    return data


def _sec_parse_form4_xml(xml_text: str) -> tuple:
    """
    Parse Form 4 primary_doc.xml. Returns (buy_count, sell_count, all_transactions).

    Counts every transactionCode of P or S across BOTH nonDerivativeTransaction
    and derivativeTransaction blocks. Codes other than P/S are ignored.
    """
    import re
    # Use regex over XML parsing because Form 4 XML namespaces are
    # inconsistent across filings (some have default ns, some don't,
    # some use ownership: prefix). Regex on <transactionCode> is robust.
    codes = re.findall(
        r'<transactionCode[^>]*>\s*([A-Z])\s*</transactionCode>',
        xml_text
    )
    buy_count  = sum(1 for c in codes if c in SEC_BUY_CODES)
    sell_count = sum(1 for c in codes if c in SEC_SELL_CODES)
    return buy_count, sell_count, codes


def _sec_fetch_and_parse_form4(cik: str, accession_no_dashes: str,
                                symbol: str, filing_date: str) -> Optional[dict]:
    """
    Fetch the primary_doc.xml for a Form 4 filing and parse its transactions.
    Cached permanently by accession_number (filings are immutable once filed).

    Returns {'buy_count': int, 'sell_count': int, 'codes': list} or None.
    """
    _sec_init_cache_db()
    # Check accession-level cache first
    conn = sqlite3.connect(SEC_FILINGS_CACHE_DB, timeout=10)
    try:
        row = conn.execute(
            'SELECT buy_count, sell_count, transactions_json '
            'FROM form4_transactions WHERE accession_number = ?',
            (accession_no_dashes,)
        ).fetchone()
        if row:
            return {
                'buy_count':  row[0],
                'sell_count': row[1],
                'codes':      json.loads(row[2]) if row[2] else [],
            }
    finally:
        conn.close()

    # Cache miss — fetch primary_doc.xml.
    # SEC filings don't all use the same filename: older filings use
    # primary_doc.xml, agent-filed Form 4s often use form4.xml, others use
    # wf-form4_*.xml. We fetch the filing's index.json to find the actual
    # .xml file, then fetch that.
    cik_unpadded = str(int(cik))
    accession_no_dashes_only = accession_no_dashes.replace('-', '')
    base_url = (f'https://www.sec.gov/Archives/edgar/data/'
                f'{cik_unpadded}/{accession_no_dashes_only}')

    xml_filename = None
    try:
        idx_r = requests.get(f'{base_url}/index.json',
                              headers=SEC_HEADERS, timeout=15)
        time.sleep(SEC_RATE_DELAY)
        if idx_r.status_code != 200:
            return None
        items = idx_r.json().get('directory', {}).get('item', [])
        # Prefer 'form4.xml' or 'primary_doc.xml'; fall back to any .xml that
        # isn't an XSL transformer (those start with 'xslF' or have 'xsl' in
        # the name and are SEC's display stylesheets, not the filing data)
        xml_candidates = [it['name'] for it in items
                          if isinstance(it, dict) and it.get('name', '').endswith('.xml')
                          and 'xsl' not in it.get('name', '').lower()]
        # Prefer known stable names
        for preferred in ('form4.xml', 'primary_doc.xml'):
            if preferred in xml_candidates:
                xml_filename = preferred
                break
        if not xml_filename and xml_candidates:
            xml_filename = xml_candidates[0]
    except Exception as e:
        print(f'  [SEC] index.json fetch failed {symbol} {accession_no_dashes}: {e}')
        return None

    if not xml_filename:
        return None

    url = f'{base_url}/{xml_filename}'

    try:
        r = requests.get(url, headers=SEC_HEADERS, timeout=15)
        time.sleep(SEC_RATE_DELAY)
        if r.status_code != 200:
            return None
        buy, sell, codes = _sec_parse_form4_xml(r.text)
    except Exception as e:
        print(f'  [SEC] form4 XML fetch failed {symbol} {accession_no_dashes}: {e}')
        return None

    # Persist to cache
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(SEC_FILINGS_CACHE_DB, timeout=10)
    try:
        conn.execute('''
            INSERT OR REPLACE INTO form4_transactions
            (accession_number, cik, symbol, filing_date,
             buy_count, sell_count, transactions_json, parsed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (accession_no_dashes, cik, symbol.upper(), filing_date,
              buy, sell, json.dumps(codes), now_iso))
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()

    return {'buy_count': buy, 'sell_count': sell, 'codes': codes}


def _sec_aggregate_recent_form4s(symbol: str, days: int) -> dict:
    """
    Core implementation. Looks up CIK, fetches submissions, filters Form 4
    filings within lookback window, parses each for P/S transaction codes,
    aggregates results.

    Returns:
        {
          'buy_count':     int,   # P transactions across all recent Form 4s
          'sell_count':    int,   # S transactions
          'filing_count':  int,   # number of Form 4 filings examined
          'latest_date':   str,
          'cik':           str,
          'error':         str (optional),
        }
    """
    cik = _sec_lookup_cik(symbol)
    if not cik:
        return {'buy_count': 0, 'sell_count': 0, 'filing_count': 0,
                'latest_date': '', 'cik': '', 'error': 'unknown_ticker'}

    subs = _sec_fetch_submissions(symbol, cik)
    if not subs:
        return {'buy_count': 0, 'sell_count': 0, 'filing_count': 0,
                'latest_date': '', 'cik': cik, 'error': 'submissions_fetch_failed'}

    recent = subs.get('filings', {}).get('recent', {})
    forms        = recent.get('form', [])
    dates        = recent.get('filingDate', [])
    accessions   = recent.get('accessionNumber', [])

    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    total_buy  = 0
    total_sell = 0
    filing_count = 0
    latest_date  = ''
    for form, fdate, accession in zip(forms, dates, accessions):
        if form != '4':
            continue
        if fdate < cutoff:
            # Filings are returned newest first, so we can early-exit
            break
        if not latest_date:
            latest_date = fdate
        result = _sec_fetch_and_parse_form4(cik, accession, symbol, fdate)
        if result:
            total_buy  += result['buy_count']
            total_sell += result['sell_count']
            filing_count += 1

    return {
        'buy_count':    total_buy,
        'sell_count':   total_sell,
        'filing_count': filing_count,
        'latest_date':  latest_date,
        'cik':          cik,
    }


def get_sec_insider_buys(symbol: str, days: int = 14) -> list:
    """
    Returns a list of recent insider PURCHASE transactions for a ticker.
    Real-money buys only (transaction code P). Grants, exercises, gifts excluded.

    Each item: {'symbol', 'filed', 'cik', 'accession'}

    Used by score_sec_insider() to compute the 0-1 signal score.
    """
    try:
        result = _sec_aggregate_recent_form4s(symbol, days)
        if result.get('error'):
            return []
        # Reconstruct individual buys from the cache for return shape compatibility
        if result['buy_count'] == 0:
            return []
        # Pull the underlying rows from the per-accession cache for this ticker/date
        conn = sqlite3.connect(SEC_FILINGS_CACHE_DB, timeout=10)
        try:
            cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            rows = conn.execute('''
                SELECT accession_number, filing_date, buy_count
                FROM form4_transactions
                WHERE symbol = ? AND filing_date >= ? AND buy_count > 0
                ORDER BY filing_date DESC
            ''', (symbol.upper(), cutoff)).fetchall()
        finally:
            conn.close()
        return [
            {
                'symbol':    symbol,
                'filed':     row[1],
                'cik':       result['cik'],
                'accession': row[0],
                'buy_count': row[2],
            }
            for row in rows
        ]
    except Exception as e:
        print(f"  [!] SEC EDGAR error for {symbol}: {e}")
        return []


def score_sec_insider(symbol: str, days: int = 14) -> float:
    """
    Returns insider buying score 0-1 based on SEC Form 4 PURCHASE transactions.
    Higher count = higher score.

    Note: rewards purchase transactions specifically (code P), not raw filing
    count. A single Form 4 with 3 separate purchase rows counts as 3.
    """
    try:
        result = _sec_aggregate_recent_form4s(symbol, days)
        buys = result.get('buy_count', 0)
        if buys >= 3:   return 1.0
        elif buys == 2: return 0.60
        elif buys == 1: return 0.30
        return 0.0
    except Exception as e:
        print(f"  [!] score_sec_insider error for {symbol}: {e}")
        return 0.0


# ── Reddit WSB Sentiment ──────────────────────────────────────────────────────



# ── Market Cap & Float Data ───────────────────────────────────────────────────

# Market cap tier thresholds (USD)
MCAP_TIERS = {
    'mega':   500_000_000_000,   # > $500B
    'large':   50_000_000_000,   # $50B - $500B
    'mid':      5_000_000_000,   # $5B  - $50B
    'small':            0,       # < $5B
}

# Base dark pool threshold per market cap tier
DP_BASE_THRESHOLD = {
    'mega':  2_000_000,   # $2M
    'large':   500_000,   # $500k
    'mid':     150_000,   # $150k
    'small':    50_000,   # $50k
}

# Cached market cap/float data (refreshed daily)
_mcap_cache: dict = {}

def get_market_cap_tier(client, symbol: str) -> tuple[str, float, float]:
    """
    Returns (tier, market_cap, float_shares) for a symbol.
    Uses Schwab quote data — no extra API needed.
    Caches results for the session.
    """
    global _mcap_cache
    if symbol in _mcap_cache:
        return _mcap_cache[symbol]

    try:
        q = get_quote(client, symbol)
        # Schwab returns totalVolume, 52wHigh, etc.
        # Estimate market cap from last price × shares outstanding
        # Use shortable shares as float proxy if available
        last        = q.get('last', q.get('lastPrice', 0))
        shares_out  = q.get('sharesOutstanding', 0)
        float_shares = q.get('shortableShares', shares_out) or shares_out

        if last <= 0 or shares_out <= 0:
            result = ('large', 0, 0)
            _mcap_cache[symbol] = result
            return result

        mcap = last * shares_out

        if mcap >= MCAP_TIERS['mega']:
            tier = 'mega'
        elif mcap >= MCAP_TIERS['large']:
            tier = 'large'
        elif mcap >= MCAP_TIERS['mid']:
            tier = 'mid'
        else:
            tier = 'small'

        result = (tier, mcap, float_shares)
        _mcap_cache[symbol] = result
        return result

    except Exception as e:
        result = ('large', 0, 0)   # safe default
        _mcap_cache[symbol] = result
        return result


def get_dp_threshold(client, symbol: str) -> float:
    """
    Returns float-adjusted dark pool threshold for a symbol.

    Formula:
      base = DP_BASE_THRESHOLD[market_cap_tier]
      float_multiplier:
        low float  (<50M shares)  → × 0.3  (even small prints matter)
        mid float  (50-500M)      → × 1.0  (baseline)
        high float (>500M shares) → × 2.0  (need bigger prints)
    """
    tier, mcap, float_shares = get_market_cap_tier(client, symbol)
    base = DP_BASE_THRESHOLD.get(tier, 500_000)

    if float_shares > 0:
        float_m = float_shares / 1_000_000   # convert to millions
        if float_m < 50:
            float_mult = 0.3
        elif float_m > 500:
            float_mult = 2.0
        else:
            float_mult = 1.0
    else:
        float_mult = 1.0

    return round(base * float_mult)


def get_dp_thresholds_bulk(client, symbols: list) -> dict:
    """Returns {symbol: threshold} for multiple symbols."""
    return {s: get_dp_threshold(client, s) for s in symbols}


# ── SEC EDGAR Form 4 (Real Insider Buying) ───────────────────────────────────

def get_sec_form4(symbol: str, days: int = 30) -> dict:
    """
    Returns aggregated Form 4 insider activity for a ticker, with buy/sell
    distinction from parsed transaction codes.

    Net signal logic (mirrors prior behavior, now backed by real data):
      net = buys - sells   (P transactions minus S transactions)
      net >= 3   →  'strong_buy', score=1.00
      net == 2   →  'buy',        score=0.75
      net == 1   →  'buy',        score=0.50
      net == 0   →  'neutral',    score=0.25
      net <  0   →  'sell',       score=0.00

    Returns:
        {
          'buy_count':    int,
          'sell_count':   int,
          'net_signal':   str,
          'score':        float,
          'filing_count': int,
          'latest_date':  str,
          'cik':          str,
        }
    """
    try:
        result = _sec_aggregate_recent_form4s(symbol, days)
        buys  = result.get('buy_count', 0)
        sells = result.get('sell_count', 0)
        net = buys - sells

        if net >= 3:   signal, score = 'strong_buy', 1.00
        elif net == 2: signal, score = 'buy',        0.75
        elif net == 1: signal, score = 'buy',        0.50
        elif net == 0: signal, score = 'neutral',    0.25
        else:          signal, score = 'sell',       0.00

        return {
            'score':        score,
            'net_signal':   signal,
            'buy_count':    buys,
            'sell_count':   sells,
            'filing_count': result.get('filing_count', 0),
            'latest_date':  result.get('latest_date', ''),
            'cik':          result.get('cik', ''),
        }
    except Exception as e:
        return {'score': 0.0, 'net_signal': 'neutral', 'buy_count': 0,
                'sell_count': 0, 'filing_count': 0, 'error': str(e)}


# ── Reddit WSB Sentiment ──────────────────────────────────────────────────────

def get_reddit_wsb_sentiment(symbol: str) -> dict:
    """
    Fetches Reddit WallStreetBets sentiment for a symbol.
    Uses Reddit's free public JSON API — no key required.

    Contrarian signal logic:
      High bearish mentions  → potential buy (crowd is wrong)
      High bullish mentions  → caution (crowded trade)
      Low/no mentions        → neutral

    Returns:
        {
          'mention_count': int,
          'bull_ratio':    float,  # 0-1
          'sentiment':     str,    # 'contrarian_buy'|'crowded'|'bullish'|'bearish'|'neutral'
          'score':         float,  # 0-1
        }
    """
    try:
        headers = {'User-Agent': 'trading-bot/2.0'}
        url     = 'https://www.reddit.com/r/wallstreetbets/search.json'
        params  = {'q': symbol, 'sort': 'new', 'limit': 50, 't': 'day'}

        r = requests.get(url, headers=headers, params=params, timeout=8)
        if r.status_code != 200:
            return {'mention_count': 0, 'sentiment': 'neutral', 'score': 0.5, 'bull_ratio': 0.5}

        posts    = r.json().get('data', {}).get('children', [])
        mentions = 0
        bullish  = 0
        bearish  = 0

        bull_words = {'calls', 'moon', 'buy', 'long', 'bullish', 'squeeze',
                      'yolo', 'rocket', 'bull', 'green', 'pump'}
        bear_words = {'puts', 'short', 'bearish', 'dump', 'crash', 'bear',
                      'red', 'sell', 'fade', 'drill'}

        for post in posts:
            data  = post.get('data', {})
            title = (data.get('title','') + ' ' + data.get('selftext','')).lower()
            sym_l = symbol.lower()

            if sym_l in title or f'${sym_l}' in title:
                mentions += 1
                b = sum(1 for w in bull_words if w in title)
                s = sum(1 for w in bear_words if w in title)
                if b > s:   bullish += 1
                elif s > b: bearish += 1

        if mentions == 0:
            return {'mention_count': 0, 'sentiment': 'neutral', 'score': 0.5, 'bull_ratio': 0.5}

        bull_ratio = bullish / mentions

        # Contrarian scoring
        if bull_ratio < 0.25 and mentions >= 3:
            sentiment, score = 'contrarian_buy', 0.80  # crowd is bearish = buy signal
        elif bull_ratio > 0.75 and mentions >= 5:
            sentiment, score = 'crowded',        0.30  # crowded long = fade signal
        elif bull_ratio >= 0.5:
            sentiment, score = 'bullish',        0.60
        elif mentions >= 2:
            sentiment, score = 'bearish',        0.40
        else:
            sentiment, score = 'neutral',        0.50

        return {
            'mention_count': mentions,
            'bull_ratio':    round(bull_ratio, 2),
            'bullish':       bullish,
            'bearish':       bearish,
            'sentiment':     sentiment,
            'score':         round(score, 2),
        }

    except Exception as e:
        return {'mention_count': 0, 'sentiment': 'neutral', 'score': 0.5,
                'bull_ratio': 0.5, 'error': str(e)}


# ── Congressional Trading (House + Senate Stock Watcher — Free) ──────────────
# Sources:
#   House: S3 bulk JSON (updated when available, historical)
#   Senate: senatestockwatcher.com/api (live, no auth required)
#
# Key insight from research: cluster signals beat individual trades.
# 3+ members buying the same ticker within 14 days = strong signal.
# Sales from multiple members = leading indicator for bad news.
# 45-day filing window means recent fast-filers are most actionable.

_congress_cache: dict = {}   # session cache to avoid re-fetching bulk files
_congress_cache_ts: float = 0.0

def _fetch_congress_bulk() -> list:
    """
    Fetches congressional trade data via Unusual Whales API.
    Endpoint: /api/congress/recent-trades
    Falls back to empty list if unavailable.
    Cache: 4 hours
    """
    global _congress_cache, _congress_cache_ts

    # Re-use cache for 4 hours
    if _congress_cache and (time.time() - _congress_cache_ts) < 14400:
        return _congress_cache.get('trades', [])

    all_trades = []

    try:
        token = os.environ.get('UW_API_KEY', '')
        if not token:
            return []

        # UW returns up to 500 recent trades — paginate to get 90 days worth
        url = 'https://api.unusualwhales.com/api/congress/recent-trades'
        headers = {'Authorization': f'Bearer {token}'}
        r = requests.get(url, headers=headers, timeout=15)

        if r.status_code == 200:
            data = r.json().get('data', [])
            for tx in data:
                # Normalize fields to match existing get_congress_trades() logic
                all_trades.append({
                    'ticker':           tx.get('ticker', ''),
                    'representative':   tx.get('name', ''),
                    'type':             'Purchase' if tx.get('txn_type','').lower() == 'buy' else tx.get('txn_type',''),
                    'disclosure_date':  tx.get('filed_at_date', ''),
                    'transaction_date': tx.get('transaction_date', ''),
                    'chamber':          tx.get('member_type', 'unknown'),
                    'amount':           tx.get('amounts', ''),
                })
    except Exception:
        pass

    _congress_cache    = {'trades': all_trades}
    _congress_cache_ts = time.time()
    return all_trades


def get_congress_trades(symbol: str, days: int = 90) -> dict:
    """
    Returns congressional trading signal for a symbol.
    Covers both House and Senate — free, no API key required.

    Scoring logic:
      3+ members buying within 14 days  → strong_buy  1.0  (cluster signal)
      2 members buying                  → buy         0.75
      1 member buying                   → buy         0.50
      0 net activity                    → neutral     0.25
      net sellers                       → sell        0.10

    Returns:
        {
          'score':       float,
          'net_signal':  str,
          'buy_count':   int,
          'sell_count':  int,
          'cluster_14d': int,   # buys in last 14 days (cluster signal)
          'politicians': list,
          'chambers':    list,
        }
    """
    try:
        from datetime import datetime, timedelta
        all_trades = _fetch_congress_bulk()
        if not all_trades:
            return {'score': 0.0, 'net_signal': 'no_data',
                    'buy_count': 0, 'sell_count': 0}

        cutoff_90d = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        cutoff_14d = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')

        sym_upper = symbol.upper()
        buys_90d  = []
        sells_90d = []
        buys_14d  = []

        for tx in all_trades:
            # Normalize ticker field — both APIs use 'ticker'
            ticker = str(tx.get('ticker', tx.get('asset_ticker', '')) or '').upper().strip()
            if ticker != sym_upper or ticker == '--':
                continue

            # Key off disclosure/filing date, NOT transaction date
            # Congress has up to 45 days to disclose — we only know on filing date
            # House API: disclosure_date = when filed with clerk (public)
            # Senate API: date_recieved = when Senate received disclosure (public)
            # Fall back to transaction_date only if no disclosure date available
            tx_date = (tx.get('disclosure_date') or
                       tx.get('date_recieved') or
                       tx.get('filed_at') or
                       tx.get('transaction_date', ''))
            if not tx_date or tx_date < cutoff_90d:
                continue

            tx_type = str(tx.get('type', tx.get('transaction_type', ''))).lower()
            name    = tx.get('representative', tx.get('name', 'Unknown'))

            if 'purchase' in tx_type or 'buy' in tx_type:
                buys_90d.append({'name': name, 'date': tx_date,
                                  'chamber': tx.get('chamber', 'unknown')})
                if tx_date >= cutoff_14d:
                    buys_14d.append(name)
            elif 'sale' in tx_type or 'sell' in tx_type or 'exchange' in tx_type:
                sells_90d.append({'name': name, 'date': tx_date})

        cluster = len(set(buys_14d))   # unique members buying in 14d
        net     = len(buys_90d) - len(sells_90d)

        if cluster >= 3:   signal, score = 'strong_buy', 1.00
        elif cluster == 2: signal, score = 'buy',        0.80
        elif net >= 2:     signal, score = 'buy',        0.65
        elif net == 1:     signal, score = 'buy',        0.50
        elif net == 0:     signal, score = 'neutral',    0.25
        else:              signal, score = 'sell',       0.10

        politicians = list({t['name'] for t in buys_90d})[:4]
        chambers    = list({t.get('chamber','') for t in buys_90d})

        return {
            'score':       score,
            'net_signal':  signal,
            'buy_count':   len(buys_90d),
            'sell_count':  len(sells_90d),
            'cluster_14d': cluster,
            'politicians': politicians,
            'chambers':    chambers,
        }

    except Exception as e:
        return {'score': 0.0, 'net_signal': 'error',
                'buy_count': 0, 'sell_count': 0, 'error': str(e)}

# ── Smoke Test ────────────────────────────────────────────────────────────────


# ── Top-of-funnel scanners ────────────────────────────────────────────────────



def get_earnings_surprise_tickers(days_back: int = 5,
                                   min_surprise_pct: float = 5.0) -> list:
    """
    Returns tickers with significant EPS beat or miss in the last days_back days.
    Uses UW /api/earnings/{ticker} — already paid for.

    PEAD effect: market underreacts to earnings surprises initially.
    Significant beat/miss = inject into scanner universe for scoring.

    Args:
        days_back:         How many days back to look for earnings reports
        min_surprise_pct:  Minimum absolute % surprise to qualify (default 5%)

    Returns:
        List of dicts: [{'symbol', 'source', 'surprise_pct', 'direction'}]
    """
    try:
        token   = os.environ.get('UW_API_KEY', '')
        if not token:
            return []

        from datetime import datetime, timedelta
        cutoff  = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        headers = {'Authorization': f'Bearer {token}'}

        # Fetch past earnings day by day using the date param
        # afterhours endpoint with ?date=YYYY-MM-DD returns reporters for that date
        results = []
        seen    = set()
        from datetime import datetime as _dt, timedelta as _td

        for day_offset in range(0, days_back + 1):
            date_str = (_dt.now() - _td(days=day_offset)).strftime('%Y-%m-%d')
            try:
                url = f'{UW_BASE_URL}/earnings/afterhours'
                r   = requests.get(url, headers=headers,
                                   params={'date': date_str}, timeout=12)
                if r.status_code != 200:
                    continue
                day_data = r.json().get('data', [])
                for item in day_data:
                    sym = (item.get('symbol') or item.get('ticker') or '').upper()
                    if not sym or sym in seen:
                        continue
                    eps_est    = item.get('street_mean_est')
                    eps_actual = item.get('actual_eps')
                    report_date = item.get('report_date', date_str)
                    sector     = item.get('sector', '')

                    if eps_est is None or eps_actual is None:
                        continue
                    try:
                        est_f = float(eps_est)
                        act_f = float(eps_actual)
                    except (TypeError, ValueError):
                        continue
                    if est_f == 0:
                        continue

                    surprise_pct = (act_f - est_f) / abs(est_f) * 100
                    # ── Quality filters (all must pass) ──────────────────
                    # 1. Cap surprise at ±100% — eliminates broken EPS calcs
                    #    where near-zero estimate produces absurd percentages
                    if abs(surprise_pct) > 100.0:
                        continue
                    # 2. Minimum surprise threshold
                    if abs(surprise_pct) < min_surprise_pct:
                        continue
                    # 3. Must have options — proxy for liquidity and size
                    if not item.get('has_options', False):
                        continue
                    # 4. Minimum market cap $5B — ensures UW has flow/dark pool data
                    #    Below this threshold, options flow signals are too sparse
                    marketcap = float(item.get('marketcap') or 0)
                    if marketcap < 1_000_000_000:
                        continue
                    # 5. Russell 3000 proxy — marketcap $500M+ with options
                    #    Russell 3000 covers ~98% of US market cap down to ~$500M
                    #    Combined with has_options and $5B floor above, this catches
                    #    mid/large caps with real options flow data in UW
                    #    (No direct Russell 3000 membership flag in UW API)
                    if marketcap < 500_000_000:
                        continue
                    seen.add(sym)
                    direction = 'bullish' if surprise_pct > 0 else 'bearish'
                    results.append({
                        'symbol':       sym,
                        'source':       'earnings_surprise',
                        'surprise_pct': round(surprise_pct, 2),
                        'direction':    direction,
                        'report_date':  report_date,
                        'marketcap':    marketcap,
                        'sector':       sector,
                    })
            except Exception:
                continue

        if results:
            print(f"  [earnings_scanner] Found {len(results)} surprise tickers "
                  f"(≥{min_surprise_pct}% surprise, last {days_back}d)")
        return results

    except Exception as e:
        print(f"  [earnings_scanner] Error: {e}")
        return []


def get_stock_ivr(symbol: str) -> float:
    """
    Fetches IV Rank (IVR) for a symbol from Unusual Whales.
    IVR = where current IV sits relative to its 52-week range (0-100).
    High IVR (>50) = elevated premium — good for selling puts (CSP).
    Low IVR (<30)  = cheap premium — poor time to sell puts.

    Returns float 0-100, or 50.0 as neutral default on failure.
    """
    try:
        token = os.environ.get('UW_API_KEY', '')
        if not token:
            return 50.0

        url     = f'{UW_BASE_URL}/stock/{symbol}/iv-rank'
        headers = {'Authorization': f'Bearer {token}'}
        r       = requests.get(url, headers=headers, timeout=8)

        if r.status_code != 200:
            return 50.0

        data = r.json().get('data', [])
        if isinstance(data, list) and data:
            # Most recent entry is first
            latest = data[0]
            ivr = latest.get('iv_rank_1y') or latest.get('iv_rank') or latest.get('ivr')
            if ivr is not None:
                return round(float(ivr), 2)

        return 50.0

    except Exception:
        return 50.0


def get_politician_tickers(days: int = 14, min_buy_count: int = 1,
                            min_market_cap_b: float = 2.0,
                            min_avg_volume: int = 500_000) -> list:
    """
    Top-of-funnel politician scanner.
    Returns tickers from recent congressional buys that pass liquidity gates,
    even if they are NOT in the standard 260-symbol universe.

    Liquidity gates (to filter illiquid names):
        min_market_cap_b:  minimum market cap in billions (default $2B)
        min_avg_volume:    minimum 30-day avg daily volume (default 500k shares)

    Returns list of dicts:
        [{'symbol': str, 'source': 'politician', 'buy_count': int,
          'politicians': list, 'cluster_14d': int}]
    """
    try:
        all_trades = _fetch_congress_bulk()
        if not all_trades:
            return []

        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        # Group buys by symbol
        buys_by_symbol = {}
        for t in all_trades:
            sym  = (t.get('ticker') or '').upper().strip()
            date = t.get('disclosure_date') or t.get('date_recieved') or ''
            txn  = t.get('type', '').lower()
            if not sym or not date or ('buy' not in txn and 'purchase' not in txn):
                continue
            if date < cutoff:
                continue
            if sym not in buys_by_symbol:
                buys_by_symbol[sym] = {'buy_count': 0, 'politicians': [], 'cluster_14d': 0}
            buys_by_symbol[sym]['buy_count'] += 1
            name = t.get('representative') or t.get('name', '')
            if name and name not in buys_by_symbol[sym]['politicians']:
                buys_by_symbol[sym]['politicians'].append(name)

        # Filter by buy count
        candidates = [
            {'symbol': sym, 'source': 'politician',
             'buy_count': v['buy_count'], 'politicians': v['politicians']}
            for sym, v in buys_by_symbol.items()
            if v['buy_count'] >= min_buy_count
        ]

        # Apply liquidity gate via yfinance quick check
        try:
            import yfinance as yf
            liquid = []
            for c in candidates:
                try:
                    info = yf.Ticker(c['symbol']).fast_info
                    mktcap = getattr(info, 'market_cap', 0) or 0
                    volume = getattr(info, 'three_month_average_volume', 0) or 0
                    if mktcap >= min_market_cap_b * 1e9 and volume >= min_avg_volume:
                        liquid.append(c)
                except Exception:
                    pass  # skip if can't verify
            candidates = liquid
        except ImportError:
            pass  # yfinance not available, return unfiltered

        return sorted(candidates, key=lambda x: x['buy_count'], reverse=True)

    except Exception as e:
        print(f'  [politician_scanner] Error: {e}')
        return []


def get_wsb_tickers(min_mentions: int = 10,
                     min_volume_surge: float = 2.0) -> list:
    """
    Top-of-funnel WSB scanner.
    Returns tickers trending on r/wallstreetbets with volume confirmation.
    Used for universe injection only — NOT for scoring weight.

    Gates:
        min_mentions:      minimum Reddit mentions in last 24h (default 10)
        min_volume_surge:  minimum today's volume / 30d avg volume (default 2x)

    Returns list of dicts:
        [{'symbol': str, 'source': 'wsb', 'mentions': int, 'volume_surge': float}]
    """
    try:
        import requests as _req
        # Pull hot posts from WSB
        headers = {'User-Agent': 'TradingBot/1.0 (research)'}
        resp = _req.get(
            'https://www.reddit.com/r/wallstreetbets/hot.json?limit=100',
            headers=headers, timeout=10
        )
        if resp.status_code != 200:
            return []

        posts = resp.json().get('data', {}).get('children', [])

        # Extract tickers (simple heuristic: 2-5 uppercase letters in title)
        import re
        ticker_mentions = {}
        pattern = re.compile(r'([A-Z]{2,5})')
        SKIP = {'WSB', 'DD', 'YOLO', 'ATH', 'CEO', 'SEC', 'FDA', 'IPO',
                'ETF', 'FED', 'GDP', 'CPI', 'EPS', 'USA', 'USD', 'EUR',
                'THE', 'FOR', 'ARE', 'NOT', 'BUT', 'YOU', 'CAN', 'HAS'}

        for post in posts:
            title = post.get('data', {}).get('title', '')
            tickers = pattern.findall(title)
            for t in tickers:
                if t not in SKIP and len(t) >= 2:
                    ticker_mentions[t] = ticker_mentions.get(t, 0) + 1

        # Filter by mentions
        candidates = [
            {'symbol': sym, 'source': 'wsb', 'mentions': count, 'volume_surge': 0.0}
            for sym, count in ticker_mentions.items()
            if count >= min_mentions
        ]

        # Volume surge confirmation via yfinance
        try:
            import yfinance as yf
            confirmed = []
            for c in candidates:
                try:
                    hist = yf.Ticker(c['symbol']).history(period='31d')
                    if len(hist) < 2:
                        continue
                    avg_vol = hist['Volume'].iloc[:-1].mean()
                    today_vol = hist['Volume'].iloc[-1]
                    surge = today_vol / avg_vol if avg_vol > 0 else 0
                    if surge >= min_volume_surge:
                        c['volume_surge'] = round(surge, 1)
                        confirmed.append(c)
                except Exception:
                    pass
            candidates = confirmed
        except ImportError:
            pass

        return sorted(candidates, key=lambda x: x['mentions'], reverse=True)

    except Exception as e:
        print(f'  [wsb_scanner] Error: {e}')
        return []

if __name__ == '__main__':
    print("🔌 Authenticating...")
    client, paper = authenticate()
    mode = "PAPER" if paper else "LIVE"
    print(f"✅ Connected ({mode} mode)\n")

    test_symbols = ['SPY', 'QQQ', 'AAPL']

    print(f"📊 VIXY: {get_vix(client):.2f}")

    print("\n📈 Quotes:")
    for sym in test_symbols:
        q = get_quote(client, sym)
        print(f"  {sym}: ${q.get('last', 0):.2f}")

    print("\n📉 Price history (last 5 candles for SPY):")
    spy_hist = get_price_history(client, 'SPY', days=10)
    if not spy_hist.empty:
        print(spy_hist.tail(5).to_string(index=False))

    print("\n🌊 UW Options Flow (SPY, last 5):")
    spy_flow = get_uw_flow('SPY', limit=5)
    if not spy_flow.empty:
        print(spy_flow[['symbol','type','side','premium','alert_type','timestamp']].to_string(index=False))
    else:
        print("  No flow data returned.")

    print("\n🏦 UW Dark Pool (last 5):")
    dp = get_uw_dark_pool(limit=5)
    if not dp.empty:
        print(dp[['symbol','price','size','premium','timestamp']].to_string(index=False))
    else:
        print("  No dark pool data returned.")

    print("\n😱 Fear & Greed Index:")
    fg = get_fear_greed()
    print(f"  Score: {fg['score']} — {fg['label']}")
    print(f"  Modifier: {fear_greed_modifier(fg['score'])}x")

    print("\n📊 Put/Call Ratio (from SPY flow):")
    if not spy_flow.empty:
        pc = compute_put_call_ratio(spy_flow)
        print(f"  Ratio: {pc['ratio']}  Signal: {pc['signal']}")
        print(f"  Calls: {pc['calls']}  Puts: {pc['puts']}")

    print("\n📐 Alpha Vantage RSI (SPY):")
    rsi = get_av_rsi('SPY')
    print(f"  RSI(14): {rsi}")

    print("\n📐 Alpha Vantage MACD (SPY):")
    macd = get_av_macd('SPY')
    print(f"  MACD: {macd}")

    print("\n📅 Finnhub Earnings (AAPL, next 7 days):")
    earnings = get_finnhub_earnings('AAPL', days_ahead=7)
    if earnings:
        for e in earnings:
            print(f"  {e.get('symbol')} — {e.get('date')} ({e.get('hour', '')})")
    else:
        print("  No earnings in next 7 days.")

    blocked, date = has_earnings_soon('AAPL', hours=48)
    print(f"  Earnings within 48h: {blocked} {date}")