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
DEFAULT_UNIVERSE = [
    'SPY', 'QQQ', 'IWM', 'DIA',
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI',
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL',
    'META', 'TSLA', 'JPM', 'GS', 'BAC',
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

        resp = client.get_price_history(
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
        resp  = client.get_quote(symbol)
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
    """Batch quote fetch. Returns {symbol: quote_dict}."""
    results = {}
    for symbol in symbols:
        results[symbol] = get_quote(client, symbol)
        time.sleep(0.1)
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

        resp = client.get_price_history(
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
        price_history[sym] = get_price_history(client, sym, days=60)
        time.sleep(0.1)

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