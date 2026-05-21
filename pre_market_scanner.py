"""
pre_market_scanner.py
Runs once daily at 6:00am ET in pre-market.
Scans 150 curated high-flow symbols using all 9 signals.
Saves a ranked watchlist to watchlist.json for intraday cycles to consume.

The main.py trading cycles read from watchlist.json instead of re-scanning,
making intraday cycles fast (~30 seconds vs ~10 minutes).

Schedule:
  6:00am ET  → this script runs, saves watchlist.json
  9:30am ET  → main.py reads watchlist, runs risk checks, trades top candidates
  1:30pm ET  → main.py reads watchlist, checks for new entries
  3:30pm ET  → main.py runs exit checks only

Run manually:   python pre_market_scanner.py
Run via cron:   0 11 * * 1-5 /root/trading-bot/venv/bin/python /root/trading-bot/pre_market_scanner.py
                (11:00 UTC = 6:00am ET, weekdays only)
"""

import os
import json
import time
import pandas as pd
from datetime import datetime, timezone
import pytz

from auth import authenticate
from notifier import send_scanner_summary
from data_collector import (
    get_price_history, get_quote, get_quotes,
    get_vix, get_vix_history, get_uw_flow,
    get_uw_dark_pool, get_fear_greed, fear_greed_modifier,
    compute_put_call_ratio, get_av_rsi, get_av_macd,
    get_politician_tickers, get_wsb_tickers, get_earnings_surprise_tickers,
)
from regime_engine import evaluate_regime
from flow_momentum import score_stock

ET = pytz.timezone('America/New_York')
WATCHLIST_FILE  = 'watchlist.json'
SCAN_START_HOUR = 6    # 6:00am ET
SCAN_END_HOUR   = 9    # must finish before 9:30am open

# ── Curated Universe (150 symbols) ───────────────────────────────────────────
# Selected for: high UW options flow, institutional activity, liquidity
# Categories: Mega cap, Large cap growth, Financials, Energy, Health,
#             Semis, China ADRs, High-vol, Sector ETFs

SCAN_UNIVERSE = [
    # ── Mega Cap Tech ──────────────────────────────────────────────────────
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'META', 'AMZN', 'TSLA',
    'NFLX', 'CRM', 'ORCL', 'ADBE', 'NOW',

    # ── Semis ──────────────────────────────────────────────────────────────
    'AMD', 'INTC', 'QCOM', 'AVGO', 'MU', 'AMAT', 'LRCX', 'KLAC',
    'ASML', 'TSM', 'ARM', 'MRVL', 'ON', 'WOLF',

    # ── Financials ─────────────────────────────────────────────────────────
    'JPM', 'GS', 'MS', 'BAC', 'WFC', 'C', 'BLK', 'SCHW',
    'V', 'MA', 'AXP', 'COF', 'PYPL',

    # ── Energy ─────────────────────────────────────────────────────────────
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO',
    'HAL', 'DVN', 'FANG',

    # ── Health / Biotech ───────────────────────────────────────────────────
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'DHR',
    'ISRG', 'REGN', 'VRTX', 'BIIB', 'MRNA', 'GILD',

    # ── Consumer / Retail ──────────────────────────────────────────────────
    'HD', 'LOW', 'TGT', 'WMT', 'COST', 'NKE', 'SBUX',
    'MCD', 'YUM', 'BKNG', 'ABNB', 'UBER', 'LYFT',

    # ── Industrials / Defense ──────────────────────────────────────────────
    'CAT', 'DE', 'LMT', 'RTX', 'NOC', 'GD', 'BA', 'GE',
    'HON', 'MMM', 'UPS', 'FDX',

    # ── High Volatility / Meme ─────────────────────────────────────────────
    'MSTR', 'COIN', 'HOOD', 'PLTR', 'RBLX', 'RIVN', 'LCID',
    'GME', 'AMC', 'SOFI', 'OPEN',

    # ── China ADRs ─────────────────────────────────────────────────────────
    'BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'XPEV', 'LI',

    # ── Broad Market ETFs ──────────────────────────────────────────────────
    'SPY', 'QQQ', 'IWM', 'DIA', 'MDY', 'VTI',

    # ── Sector ETFs ────────────────────────────────────────────────────────
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLB', 'XLU', 'XLRE',
    'XLC', 'XLY', 'XLP', 'GDX', 'GDXJ',

    # ── Volatility / Inverse ───────────────────────────────────────────────
    'VIXY', 'VXX', 'SQQQ', 'SPXU', 'UVXY',

    # ── Fixed Income / Macro ───────────────────────────────────────────────
    'TLT', 'HYG', 'LQD', 'GLD', 'SLV', 'USO', 'UNG',
]

# Deduplicate while preserving order
seen = set()
SCAN_UNIVERSE = [s for s in SCAN_UNIVERSE if not (s in seen or seen.add(s))]
print(f"Universe: {len(SCAN_UNIVERSE)} symbols")


# ── Watchlist Schema ──────────────────────────────────────────────────────────

def build_watchlist_entry(stock_score, quote: dict, rsi: float, macd: dict) -> dict:
    return {
        'symbol':      stock_score.symbol,
        'score':       stock_score.total_score,
        'direction':   stock_score.direction,
        'qualifies':   stock_score.qualifies,
        'signals':     stock_score.signals,
        'weighted':    stock_score.weighted,
        'last_price':  quote.get('last', 0),
        'volume':      quote.get('volume', 0),
        'rsi':         rsi,
        'macd_hist':   macd.get('histogram', 0),
        'macd_signal': 'bullish' if macd.get('histogram', 0) > 0 else 'bearish',
        'scanned_at':  datetime.now(ET).isoformat(),
    }


def save_watchlist(entries: list, regime: str, vix: float, fg: dict, pc: dict):
    watchlist = {
        'date':          datetime.now(ET).strftime('%Y-%m-%d'),
        'scanned_at':    datetime.now(ET).isoformat(),
        'regime':        regime,
        'vixy':          vix,
        'fear_greed':    fg,
        'put_call':      pc,
        'symbols_scanned': len(entries),
        'qualified':     len([e for e in entries if e['qualifies']]),
        'entries':       sorted(entries, key=lambda x: x['score'], reverse=True),
    }
    with open(WATCHLIST_FILE, 'w') as f:
        json.dump(watchlist, f, indent=2)
    print(f"  💾 Watchlist saved: {WATCHLIST_FILE}")
    return watchlist


def load_watchlist() -> dict:
    """Load today's watchlist. Returns None if stale or missing."""
    if not os.path.exists(WATCHLIST_FILE):
        return None
    with open(WATCHLIST_FILE) as f:
        wl = json.load(f)
    today = datetime.now(ET).strftime('%Y-%m-%d')
    if wl.get('date') != today:
        print(f"  ⚠️  Watchlist is from {wl.get('date')} — stale, re-scan needed")
        return None
    return wl


def get_watchlist_candidates(min_score: float = 0.75, max_n: int = 20) -> list:
    """
    Returns top N qualified candidates from today's watchlist.
    Used by main.py intraday cycles instead of re-scanning.
    """
    wl = load_watchlist()
    if not wl:
        return []
    candidates = [
        e for e in wl['entries']
        if e['score'] >= min_score and e['direction'] != 'bearish'
    ]
    return candidates[:max_n]


# ── Main Scanner ──────────────────────────────────────────────────────────────

def run_scan(client) -> dict:
    scan_start = datetime.now(ET)
    print(f"\n{'='*62}")
    print(f"  PRE-MARKET SCANNER")
    print(f"  {scan_start.strftime('%Y-%m-%d %H:%M:%S ET')}")
    print(f"  Universe: {len(SCAN_UNIVERSE)} symbols")
    print(f"{'='*62}\n")

    # ── Core data ──────────────────────────────────────────────────────────
    print("📡 Fetching core market data...")
    vixy        = get_vix(client)
    vix_history = get_vix_history(client, days=30)
    regime      = evaluate_regime(vixy, vix_history)
    print(f"  VIXY: {vixy:.2f}  Regime: {regime.regime.upper()}")

    fg    = get_fear_greed()
    fg_mod = fear_greed_modifier(fg.get('score', 50))
    print(f"  Fear & Greed: {fg.get('score', 50)} ({fg.get('label', 'N/A')}) → {fg_mod:.2f}x")

    # ── Fetch SPY for market tide ───────────────────────────────────────────
    print("\n📈 Fetching SPY history...")
    spy_hist = get_price_history(client, 'SPY', days=60)

    # ── Batch quotes for all symbols ───────────────────────────────────────
    print(f"\n📊 Fetching quotes for {len(SCAN_UNIVERSE)} symbols...")
    quotes = {}
    batch_size = 20
    for i in range(0, len(SCAN_UNIVERSE), batch_size):
        batch = SCAN_UNIVERSE[i:i+batch_size]
        batch_quotes = get_quotes(client, batch)
        quotes.update(batch_quotes)
        print(f"  Quotes: {min(i+batch_size, len(SCAN_UNIVERSE))}/{len(SCAN_UNIVERSE)}")
        time.sleep(0.5)

    # ── Fetch UW market-wide flow ───────────────────────────────────────────
    print("\n🌊 Fetching UW market-wide flow...")
    flow_frames = []
    for sym in SCAN_UNIVERSE[:50]:   # top 50 by priority for flow
        df = get_uw_flow(sym, limit=30)
        if not df.empty:
            flow_frames.append(df)
        time.sleep(0.25)
    combined_flow = pd.concat(flow_frames, ignore_index=True) if flow_frames else pd.DataFrame()

    print("\n🏦 Fetching dark pool prints...")
    dp_df = get_uw_dark_pool(limit=200)

    # Put/call ratio
    pc = compute_put_call_ratio(combined_flow)
    pc_mod = 1.15 if pc['signal'] == 'extreme_fear' else \
             1.05 if pc['signal'] == 'fear' else \
             0.90 if pc['signal'] == 'extreme_greed' else 1.0
    print(f"  P/C ratio: {pc['ratio']:.2f} ({pc['signal']}) → {pc_mod:.2f}x")

    # ── Top-of-funnel injection ────────────────────────────────────────────
    print("\n🏛️  Scanning politician + WSB top-of-funnel...")
    scan_universe = list(SCAN_UNIVERSE)
    injected = {}

    try:
        pol_tickers = get_politician_tickers(days=90, min_buy_count=1)
        for t in pol_tickers:
            sym = t['symbol']
            if sym not in scan_universe:
                scan_universe.append(sym)
                injected[sym] = 'politician'
                print(f"  [top-funnel] POLITICIAN injected: {sym} ({t['buy_count']} buy(s))")
    except Exception as e:
        print(f"  [top-funnel] Politician scan error: {e}")

    try:
        wsb_tickers = get_wsb_tickers(min_mentions=5, min_volume_surge=1.5)
        for t in wsb_tickers:
            sym = t['symbol']
            if sym not in scan_universe and sym not in injected:
                scan_universe.append(sym)
                injected[sym] = 'wsb'
                print(f"  [top-funnel] WSB injected: {sym} ({t['mentions']} mentions, {t['volume_surge']}x vol)")
    except Exception as e:
        print(f"  [top-funnel] WSB scan error: {e}")

    try:
        earn_tickers = get_earnings_surprise_tickers(days_back=5, min_surprise_pct=5.0)
        for t in earn_tickers:
            sym = t['symbol']
            if sym not in scan_universe:
                scan_universe.append(sym)
            if sym not in injected:
                injected[sym] = 'earnings_surprise'
                direction = t.get('direction', '')
                surprise  = t.get('surprise_pct', 0)
                sign      = '+' if surprise > 0 else ''
                print(f"  [top-funnel] EARNINGS injected: {sym} ({sign}{surprise:.1f}% surprise, {direction})")
    except Exception as e:
        print(f"  [top-funnel] Earnings scan error: {e}")

    if injected:
        print(f"  Injected {len(injected)} symbols: {list(injected.keys())}")
    else:
        print("  No injections this cycle")

    # ── Score all symbols ──────────────────────────────────────────────────
    print(f"\n🔍 Scoring {len(scan_universe)} symbols ({len(SCAN_UNIVERSE)} base + {len(injected)} injected)...")
    entries     = []
    price_cache = {}
    scored      = 0

    for symbol in scan_universe:
        try:
            # Fetch price history (cache to avoid duplicate calls)
            if symbol not in price_cache:
                price_cache[symbol] = get_price_history(client, symbol, days=60)
                time.sleep(0.1)

            price_history = {symbol: price_cache[symbol], 'SPY': spy_hist}

            # Full 9-signal score
            stock_score = score_stock(
                symbol=symbol,
                uw_flow_df=combined_flow,
                dp_df=dp_df,
                price_history=price_history,
                quotes=quotes,
                spy_history=spy_hist,
            )

            # Apply Fear & Greed and P/C modifiers
            stock_score.total_score = round(
                min(stock_score.total_score * fg_mod * pc_mod, 1.0), 4
            )
            stock_score.qualifies = stock_score.total_score >= 0.80  # sync with MIN_SCORE

            # Get RSI and MACD (rate limited — only for high scorers)
            rsi  = 50.0
            macd = {}
            if stock_score.total_score >= 0.40:
                rsi  = get_av_rsi(symbol)
                macd = get_av_macd(symbol)
                time.sleep(0.5)   # AV free tier rate limit

            quote  = quotes.get(symbol, {})
            entry  = build_watchlist_entry(stock_score, quote, rsi, macd)
            entry['source'] = injected.get(symbol, 'universe')
            entries.append(entry)
            scored += 1

            if scored % 25 == 0:
                print(f"  Scored {scored}/{len(SCAN_UNIVERSE)}...")

        except Exception as e:
            print(f"  ⚠️  Error scoring {symbol}: {e}")
            continue

    # ── Save watchlist ─────────────────────────────────────────────────────
    watchlist = save_watchlist(entries, regime.regime, vixy, fg, pc)

    # ── Print summary ──────────────────────────────────────────────────────
    elapsed = (datetime.now(ET) - scan_start).seconds // 60
    qualified = [e for e in entries if e['qualifies']]

    print(f"\n{'='*62}")
    print(f"  SCAN COMPLETE  ({elapsed} minutes)")
    print(f"{'='*62}")
    print(f"  Symbols scanned: {len(entries)} ({len(SCAN_UNIVERSE)} base + {len(injected)} injected)")
    print(f"  Qualified (≥0.80): {len(qualified)}")
    print(f"  Regime: {regime.regime.upper()}  VIXY: {vixy:.2f}")
    print(f"\n  Top 20 by score:")
    print(f"  {'Symbol':<8} {'Score':>6}  {'Dir':<10} {'RSI':>5}  {'MACD':>8}  Qualifies")
    print(f"  {'-'*55}")

    top20 = sorted(entries, key=lambda x: x['score'], reverse=True)[:20]
    for e in top20:
        qual = '✅' if e['qualifies'] else ''
        macd_str = f"{e['macd_hist']:>+.3f}" if e['macd_hist'] else '  N/A '
        print(f"  {e['symbol']:<8} {e['score']:>6.3f}  {e['direction']:<10} "
              f"{e['rsi']:>5.1f}  {macd_str}  {qual}")

    if qualified:
        print(f"\n  🎯 Qualified candidates for today:")
        for e in sorted(qualified, key=lambda x: x['score'], reverse=True):
            print(f"  {'✅'} {e['symbol']:<8} score={e['score']:.3f}  "
                  f"dir={e['direction']}  rsi={e['rsi']:.1f}  "
                  f"price=${e['last_price']:.2f}")
    else:
        print(f"\n  ⏸  No symbols qualified today — cash stays parked")

    print(f"\n  Watchlist saved → {WATCHLIST_FILE}")
    print(f"{'='*62}\n")

    # Send scanner summary email
    try:
        send_scanner_summary(watchlist)
    except Exception as e:
        print(f"  [notifier] Scanner email failed: {e}")

    return watchlist


# ── Scheduler ─────────────────────────────────────────────────────────────────

def wait_for_scan_time():
    """Wait until 6:00am ET if run before that."""
    now = datetime.now(ET)
    if now.hour < SCAN_START_HOUR:
        target = now.replace(hour=SCAN_START_HOUR, minute=0, second=0, microsecond=0)
        wait_seconds = (target - now).seconds
        print(f"⏰ Waiting until 6:00am ET ({wait_seconds//60} minutes)...")
        time.sleep(wait_seconds)


def is_weekday() -> bool:
    return datetime.now(ET).weekday() < 5


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--now', action='store_true',
                        help='Run immediately without waiting for 6am')
    parser.add_argument('--loop', action='store_true',
                        help='Run daily in a loop (for server deployment)')
    args = parser.parse_args()

    print("🔌 Authenticating...")
    client, paper = authenticate()
    paper = True  # assume paper mode
    print(f"✅ Connected ({'PAPER' if paper else 'LIVE'} mode)\n")

    if args.loop:
        print("🔄 Running in daily loop mode...")
        while True:
            if is_weekday():
                if not args.now:
                    wait_for_scan_time()
                run_scan(client)
            else:
                print("📅 Weekend — skipping scan")

            # Sleep until next morning
            now        = datetime.now(ET)
            tomorrow   = now.replace(hour=SCAN_START_HOUR, minute=0,
                                     second=0, microsecond=0)
            if tomorrow <= now:
                from datetime import timedelta as _td
                tomorrow = tomorrow + _td(days=1)
            sleep_sec  = (tomorrow - now).seconds
            print(f"💤 Sleeping until tomorrow 6:00am ET ({sleep_sec//3600:.1f}h)...")
            time.sleep(sleep_sec)
    else:
        if not args.now and not is_weekday():
            print("📅 Weekend — use --now to force scan")
        else:
            if not args.now:
                wait_for_scan_time()
            run_scan(client)