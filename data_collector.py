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

def get_sec_insider_buys(symbol: str, days: int = 14) -> list:
    """
    Fetches recent insider buying from SEC EDGAR full-text search.
    Filters for Form 4 filings (insider transactions) with buy transactions.
    Free, no API key required.
    Returns list of buy transactions with date, insider name, shares, value.
    """
    try:
        # EDGAR full-text search for Form 4 filings
        url = 'https://efts.sec.gov/LATEST/search-index'
        params = {
            'q':        f'"{symbol}"',
            'dateRange': 'custom',
            'startdt':  (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'enddt':    datetime.now().strftime('%Y-%m-%d'),
            'forms':    '4',
        }
        headers = {'User-Agent': 'trading-bot contact@example.com'}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code != 200:
            return []

        hits = r.json().get('hits', {}).get('hits', [])
        buys = []
        for hit in hits[:10]:
            src = hit.get('_source', {})
            # Only include buys (transaction code P = purchase)
            if 'P' in str(src.get('period_of_report', '')):
                continue
            display = src.get('display_date_filed', '')
            entity  = src.get('entity_name', '')
            buys.append({
                'symbol':    symbol,
                'filed':     display,
                'insider':   entity,
                'form':      src.get('form_type', '4'),
                'accession': src.get('accession_no', ''),
            })
        return buys
    except Exception as e:
        print(f"  [!] SEC EDGAR error for {symbol}: {e}")
        return []


def score_sec_insider(symbol: str, days: int = 14) -> float:
    """
    Returns insider buying score 0-1 based on SEC Form 4 filings.
    Multiple recent filings = higher score.
    """
    buys = get_sec_insider_buys(symbol, days)
    if not buys:
        return 0.0
    # Score based on number of recent insider buy filings
    if len(buys) >= 3:   return 1.0
    elif len(buys) == 2: return 0.60
    elif len(buys) == 1: return 0.30
    return 0.0


# ── Reddit WSB Sentiment ──────────────────────────────────────────────────────

def get_reddit_sentiment(symbol: str) -> dict:
    """
    Fetches Reddit WSB sentiment for a symbol using free JSON API.
    No API key required — uses Reddit's public JSON endpoint.
    Returns dict with mention_count, sentiment (bullish/bearish/neutral), score.

    Contrarian signal:
      High mentions + bearish = potential buy (crowd is wrong)
      High mentions + bullish = caution (crowd may be right but fading)
    """
    try:
        headers = {'User-Agent': 'trading-bot/1.0'}
        # Search WSB for symbol mentions in hot posts
        url = f'https://www.reddit.com/r/wallstreetbets/search.json'
        params = {
            'q':      symbol,
            'sort':   'new',
            'limit':  25,
            't':      'day',
        }
        r = requests.get(url, headers=headers, params=params, timeout=8)
        if r.status_code != 200:
            return {'mention_count': 0, 'sentiment': 'neutral', 'score': 0.5}

        posts   = r.json().get('data', {}).get('children', [])
        mentions = 0
        bullish  = 0
        bearish  = 0

        bull_words = ['calls', 'moon', 'buy', 'long', 'bullish', 'squeeze', 'yolo', '🚀', '🟢']
        bear_words = ['puts', 'short', 'bearish', 'dump', 'crash', 'puts', '🔴', '💀']

        for post in posts:
            data  = post.get('data', {})
            title = (data.get('title', '') + ' ' + data.get('selftext', '')).lower()
            if symbol.lower() in title or f'${symbol.lower()}' in title:
                mentions += 1
                bulls = sum(1 for w in bull_words if w in title)
                bears = sum(1 for w in bear_words if w in title)
                if bulls > bears:  bullish += 1
                elif bears > bulls: bearish += 1

        if mentions == 0:
            return {'mention_count': 0, 'sentiment': 'neutral', 'score': 0.5}

        bull_ratio = bullish / mentions if mentions else 0.5

        # Contrarian scoring: extreme bearish = buy signal
        if bull_ratio < 0.25 and mentions >= 3:
            sentiment = 'contrarian_buy'
            score     = 0.80
        elif bull_ratio > 0.75 and mentions >= 5:
            sentiment = 'crowded_long'
            score     = 0.30   # fade the crowd
        elif mentions >= 3:
            sentiment = 'bullish' if bull_ratio > 0.5 else 'bearish'
            score     = 0.55 if bull_ratio > 0.5 else 0.45
        else:
            sentiment = 'neutral'
            score     = 0.50

        return {
            'mention_count': mentions,
            'bullish':       bullish,
            'bearish':       bearish,
            'bull_ratio':    round(bull_ratio, 2),
            'sentiment':     sentiment,
            'score':         round(score, 2),
        }
    except Exception as e:
        print(f"  [!] Reddit sentiment error for {symbol}: {e}")
        return {'mention_count': 0, 'sentiment': 'neutral', 'score': 0.5}



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
    Fetches real insider buying from SEC EDGAR full-text search.
    Looks for Form 4 filings with Purchase (P) transaction codes.
    Free, no API key required.

    Returns:
        {
          'buy_count':    int,    # number of insider buy filings
          'sell_count':   int,    # number of insider sell filings
          'net_signal':   str,    # 'strong_buy' | 'buy' | 'neutral' | 'sell'
          'score':        float,  # 0.0 - 1.0
          'latest_date':  str,
          'insiders':     list,   # names of buyers
        }
    """
    try:
        from datetime import datetime, timedelta
        start_dt = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        end_dt   = datetime.now().strftime('%Y-%m-%d')

        url = 'https://efts.sec.gov/LATEST/search-index'
        params = {
            'q':        f'"{symbol}"',
            'dateRange': 'custom',
            'startdt':  start_dt,
            'enddt':    end_dt,
            'forms':    '4',
        }
        headers = {'User-Agent': 'trading-bot research@example.com'}
        r = requests.get(url, params=params, headers=headers, timeout=10)

        if r.status_code != 200:
            return {'score': 0.0, 'net_signal': 'neutral', 'buy_count': 0, 'sell_count': 0}

        hits     = r.json().get('hits', {}).get('hits', [])
        buys     = 0
        sells    = 0
        insiders = []

        for hit in hits[:20]:
            src   = hit.get('_source', {})
            # Use file_date (when filing became public) not period_of_report (trade date)
            # File date = public disclosure date; period_of_report = private trade date
            filed = src.get('file_date', src.get('period_of_report', ''))
            name  = src.get('display_names', [''])[0] if src.get('display_names') else ''

            # Heuristic: check for buy/sell indicators in filing text
            snippet = str(src).lower()
            if any(w in snippet for w in ['purchase', 'acquired', 'bought']):
                buys += 1
                if name: insiders.append(name)
            elif any(w in snippet for w in ['sale', 'sold', 'disposed']):
                sells += 1

        net = buys - sells
        if net >= 3:   signal, score = 'strong_buy', 1.00
        elif net == 2: signal, score = 'buy',        0.75
        elif net == 1: signal, score = 'buy',        0.50
        elif net == 0: signal, score = 'neutral',    0.25
        else:          signal, score = 'sell',       0.00

        return {
            'score':      score,
            'net_signal': signal,
            'buy_count':  buys,
            'sell_count': sells,
            'insiders':   insiders[:3],
        }

    except Exception as e:
        return {'score': 0.0, 'net_signal': 'neutral', 'buy_count': 0,
                'sell_count': 0, 'error': str(e)}


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
    Fetches and caches combined House + Senate congressional trade data.
    House: S3 bulk JSON  (last available snapshot)
    Senate: senatestockwatcher.com live JSON API
    Returns combined list of trade dicts.
    """
    global _congress_cache, _congress_cache_ts
    import time as _time

    # Re-use cache for 4 hours
    if _congress_cache and (_time.time() - _congress_cache_ts) < 14400:
        return _congress_cache.get('trades', [])

    all_trades = []

    # Senate — live API, clean JSON, no auth
    try:
        senate_url = 'https://senatestockwatcher.com/api'
        r = requests.get(senate_url, timeout=12,
                         headers={'User-Agent': 'trading-bot/2.0'})
        if r.status_code == 200:
            data = r.json()
            # Senate API returns list of senator objects with transactions array
            if isinstance(data, list):
                for senator in data:
                    txns = senator.get('transactions', [])
                    name = f"{senator.get('first_name','')} {senator.get('last_name','')}".strip()
                    for tx in txns:
                        tx['representative'] = name
                        tx['chamber']        = 'senate'
                        all_trades.append(tx)
            elif isinstance(data, dict) and 'transactions' in data:
                # Some endpoints return flat list
                for tx in data['transactions']:
                    tx['chamber'] = 'senate'
                    all_trades.append(tx)
    except Exception as e:
        pass   # Senate unavailable — fall through to house data

    # House — S3 bulk JSON (historical, updated periodically)
    try:
        house_url = ('https://house-stock-watcher-data.s3-us-west-2'
                     '.amazonaws.com/data/all_transactions.json')
        r = requests.get(house_url, timeout=15,
                         headers={'User-Agent': 'trading-bot/2.0'})
        if r.status_code == 200:
            house_trades = r.json()
            for tx in house_trades:
                tx['chamber'] = 'house'
            all_trades.extend(house_trades)
    except Exception as e:
        pass

    _congress_cache    = {'trades': all_trades}
    _congress_cache_ts = _time.time()
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
            ticker = str(tx.get('ticker', tx.get('asset_ticker', ''))).upper().strip()
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