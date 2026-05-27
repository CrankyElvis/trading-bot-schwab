"""
download_calls_history.py — MarketData.app Historical Call Options Downloader

Downloads historical covered-call pricing for tickers in the bot universe
across the 5-year backtest period. Stores in SQLite (same table as puts,
keyed by side='call') for offline wheel-strategy backtesting.

Strategy:
  For each trading day × ticker:
    1. Find the nearest expiration ~45 DTE from entry date
    2. Find the 8% OTM call strike (strike ABOVE underlying)
    3. Fetch bid/ask/mid from MarketData.app
    4. Store in SQLite with side='call'

NOTE: This is a sibling of download_options_history.py (which handles puts).
      Both write to the same options_history table — the (symbol, trade_date,
      expiration, strike, side) primary key allows put + call coexistence.

Usage:
    python download_calls_history.py                  # full 5yr download
    python download_calls_history.py --years 1        # test with 1yr
    python download_calls_history.py --ticker AAPL    # single ticker test
    python download_calls_history.py --status         # show DB status

Resume: automatically skips already-downloaded (ticker, date) pairs for CALLS
        specifically (won't be confused by existing put records).
"""

import os
import sys
import time
import sqlite3
import argparse
import requests
from datetime import datetime, timedelta, date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────

DB_PATH        = '/root/trading-bot/data/options_history.db'
API_BASE       = 'https://api.marketdata.app/v1'
TARGET_DTE     = 45     # target days to expiration
OTM_PCT        = 0.08   # 8% OTM for call strike (strike ABOVE underlying)
RATE_LIMIT_RPS = 5      # requests per second (conservative)
SLEEP_BETWEEN  = 1.0 / RATE_LIMIT_RPS
SIDE           = 'call'

sys.path.insert(0, str(Path(__file__).parent))

try:
    from data_collector import DEFAULT_UNIVERSE
    UNIVERSE = list(DEFAULT_UNIVERSE)
except Exception:
    UNIVERSE = ['AAPL','MSFT','NVDA','GOOGL','META','AMZN','TSLA','JPM',
                'AMD','SPY','QQQ','IWM']

# Add key tickers not in default universe
for t in ['SPY','QQQ','IWM','GLD','VIXY']:
    if t not in UNIVERSE:
        UNIVERSE.append(t)

# ── Database ───────────────────────────────────────────────────────────────

def init_db(db_path: str):
    """Reuse the existing options_history table — no schema change needed."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS options_history (
            symbol          TEXT NOT NULL,
            trade_date      TEXT NOT NULL,
            expiration      TEXT NOT NULL,
            strike          REAL NOT NULL,
            side            TEXT NOT NULL,
            dte             INTEGER,
            bid             REAL,
            ask             REAL,
            mid             REAL,
            volume          INTEGER,
            open_interest   INTEGER,
            underlying_price REAL,
            in_the_money    INTEGER,
            fetched_at      TEXT,
            PRIMARY KEY (symbol, trade_date, expiration, strike, side)
        )
    ''')
    # Separate error tracking by side so call errors don't block put retries
    conn.execute('''
        CREATE TABLE IF NOT EXISTS fetch_errors_calls (
            symbol     TEXT,
            trade_date TEXT,
            error      TEXT,
            fetched_at TEXT,
            PRIMARY KEY (symbol, trade_date)
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_sym_date ON options_history(symbol, trade_date)')
    conn.commit()
    return conn


def already_fetched(conn, symbol, trade_date):
    """Check if CALLS specifically have been fetched for this (symbol, date).
    Critical: must filter by side='call' since the table also holds puts."""
    cur = conn.execute(
        "SELECT 1 FROM options_history WHERE symbol=? AND trade_date=? AND side='call' LIMIT 1",
        (symbol, trade_date)
    )
    if cur.fetchone():
        return True
    cur = conn.execute(
        'SELECT 1 FROM fetch_errors_calls WHERE symbol=? AND trade_date=? LIMIT 1',
        (symbol, trade_date)
    )
    return cur.fetchone() is not None


def save_record(conn, record):
    conn.execute('''
        INSERT OR REPLACE INTO options_history
        (symbol, trade_date, expiration, strike, side, dte,
         bid, ask, mid, volume, open_interest, underlying_price,
         in_the_money, fetched_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', record)
    conn.commit()


def save_error(conn, symbol, trade_date, error):
    conn.execute('''
        INSERT OR REPLACE INTO fetch_errors_calls (symbol, trade_date, error, fetched_at)
        VALUES (?,?,?,?)
    ''', (symbol, trade_date, str(error), datetime.now().isoformat()))
    conn.commit()


# ── API ────────────────────────────────────────────────────────────────────

def get_headers():
    token = os.environ.get('MARKETDATA_API_KEY', '')
    if not token:
        raise ValueError("MARKETDATA_API_KEY not set in .env")
    return {'Authorization': f'Bearer {token}'}


def get_expirations(symbol, trade_date_str):
    url = f'{API_BASE}/options/expirations/{symbol}/'
    r = requests.get(url, headers=get_headers(),
                     params={'date': trade_date_str}, timeout=15)
    if r.status_code not in (200, 203):
        return []
    data = r.json()
    return data.get('expirations', [])


def find_target_expiration(expirations, trade_date_str, target_dte=45):
    trade_dt = datetime.strptime(trade_date_str, '%Y-%m-%d').date()
    best_exp = None
    best_diff = 9999
    for exp in expirations:
        try:
            exp_dt = datetime.strptime(exp, '%Y-%m-%d').date()
            dte = (exp_dt - trade_dt).days
            if dte < 21:
                continue
            diff = abs(dte - target_dte)
            if diff < best_diff:
                best_diff = diff
                best_exp = exp
        except Exception:
            continue
    return best_exp


# ── Trading days ───────────────────────────────────────────────────────────

def get_trading_days(years=5):
    end   = date.today() - timedelta(days=1)
    start = end - timedelta(days=years * 365)
    holidays = set()
    for yr in range(start.year, end.year + 1):
        holidays.update([
            date(yr, 1, 1),
            date(yr, 7, 4),
            date(yr, 12, 25),
        ])
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5 and d not in holidays:
            days.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    return days


# ── Main download loop ─────────────────────────────────────────────────────

def download(years=5, ticker_filter=None):
    conn = init_db(DB_PATH)
    if not os.environ.get('MARKETDATA_API_KEY'):
        print("❌ MARKETDATA_API_KEY not set")
        sys.exit(1)

    tickers      = [ticker_filter] if ticker_filter else UNIVERSE
    trade_days   = get_trading_days(years)
    total        = len(tickers) * len(trade_days)
    done         = 0
    skipped      = 0
    errors       = 0
    fetched      = 0
    credits_used = 0

    print(f"📥 CALL Options History Downloader")
    print(f"   Tickers: {len(tickers)} | Days: {len(trade_days)} | Total: {total:,}")
    print(f"   Side:    {SIDE} ({OTM_PCT*100:.0f}% OTM, ~{TARGET_DTE} DTE)")
    print(f"   DB:      {DB_PATH}")
    print(f"   Resume:  enabled (skips already-downloaded call dates)")
    print()

    t0 = time.time()

    for ticker in tickers:
        print(f"\n── {ticker} ──────────────────────────────────────")
        ticker_fetched = 0
        ticker_skipped = 0

        for trade_date in trade_days:
            done += 1

            if already_fetched(conn, ticker, trade_date):
                skipped += 1
                ticker_skipped += 1
                continue

            try:
                # Step 1: Get expirations
                time.sleep(SLEEP_BETWEEN)
                expirations = get_expirations(ticker, trade_date)
                credits_used += 1
                if not expirations:
                    save_error(conn, ticker, trade_date, 'no_expirations')
                    errors += 1
                    continue

                # Step 2: Find target expiration (~45 DTE)
                expiration = find_target_expiration(expirations, trade_date, TARGET_DTE)
                if not expiration:
                    save_error(conn, ticker, trade_date, 'no_valid_expiration')
                    errors += 1
                    continue

                # Step 3: Get the CALL chain for this expiration
                time.sleep(SLEEP_BETWEEN)
                url = f'{API_BASE}/options/chain/{ticker}/'
                r = requests.get(url, headers=get_headers(),
                                 params={'date': trade_date, 'expiration': expiration,
                                         'side': SIDE, 'minOpenInterest': 10},
                                 timeout=15)
                credits_used += 1
                if r.status_code not in (200, 203):
                    save_error(conn, ticker, trade_date, f'chain_error_{r.status_code}')
                    errors += 1
                    continue

                chain_data = r.json()
                if chain_data.get('s') != 'ok':
                    save_error(conn, ticker, trade_date, 'chain_no_data')
                    errors += 1
                    continue

                underlying_prices = chain_data.get('underlyingPrice', [])
                if not underlying_prices:
                    save_error(conn, ticker, trade_date, 'no_underlying_price')
                    errors += 1
                    continue

                underlying_price = underlying_prices[0]
                # CALL: target strike is ABOVE underlying (+8%)
                target_strike    = round(underlying_price * (1 + OTM_PCT), 0)

                # Step 4: Find nearest call strike at or near target
                strikes = chain_data.get('strike', [])
                if not strikes:
                    save_error(conn, ticker, trade_date, 'no_strikes')
                    errors += 1
                    continue

                nearest_strike = min(strikes, key=lambda s: abs(float(s) - target_strike))
                nearest_strike = float(nearest_strike)

                # Step 5: Extract quote data at the nearest strike index
                bids = chain_data.get('bid', [])
                asks = chain_data.get('ask', [])
                mids = chain_data.get('mid', [])
                dtes = chain_data.get('dte', [])
                vols = chain_data.get('volume', [])
                ois  = chain_data.get('openInterest', [])
                itms = chain_data.get('inTheMoney', [])

                try:
                    idx = [float(s) for s in strikes].index(nearest_strike)
                except ValueError:
                    diffs = [abs(float(s) - nearest_strike) for s in strikes]
                    idx = diffs.index(min(diffs))

                def safe_get(lst, i):
                    try: return lst[i]
                    except: return None

                bid = safe_get(bids, idx)
                ask = safe_get(asks, idx)
                mid = safe_get(mids, idx)
                dte = safe_get(dtes, idx)
                vol = safe_get(vols, idx)
                oi  = safe_get(ois, idx)
                itm = 1 if safe_get(itms, idx) else 0

                if mid is None:
                    save_error(conn, ticker, trade_date, 'no_mid_price')
                    errors += 1
                    continue

                # Step 6: Save with side='call'
                record = (
                    ticker, trade_date, expiration, nearest_strike, SIDE,
                    dte, bid, ask, mid, vol, oi, underlying_price, itm,
                    datetime.now().isoformat()
                )
                save_record(conn, record)
                fetched += 1
                ticker_fetched += 1
                credits_used += 1

                elapsed   = time.time() - t0
                rate      = fetched / max(elapsed, 1)
                remaining = (total - done - skipped) / max(rate, 0.01)
                print(f"  ✅ {trade_date} | exp={expiration} | "
                      f"strike={nearest_strike} | mid=${mid:.2f} | "
                      f"credits={credits_used} | "
                      f"ETA={remaining/3600:.1f}h", end='\r')

            except KeyboardInterrupt:
                print(f"\n\n⏸  Interrupted — progress saved. Re-run to continue.")
                conn.close()
                sys.exit(0)
            except Exception as e:
                save_error(conn, ticker, trade_date, str(e))
                errors += 1

        print(f"\n  {ticker}: fetched={ticker_fetched} skipped={ticker_skipped}")

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"✅ Call download complete")
    print(f"   Fetched:  {fetched:,} records")
    print(f"   Skipped:  {skipped:,} (already in DB)")
    print(f"   Errors:   {errors:,}")
    print(f"   Credits:  {credits_used:,}")
    print(f"   Time:     {elapsed:.1f} min")
    print(f"   DB:       {DB_PATH}")
    conn.close()


def show_status():
    if not os.path.exists(DB_PATH):
        print("❌ Database not found")
        return
    conn = sqlite3.connect(DB_PATH)
    call_records = conn.execute(
        "SELECT COUNT(*) FROM options_history WHERE side='call'").fetchone()[0]
    put_records = conn.execute(
        "SELECT COUNT(*) FROM options_history WHERE side='put'").fetchone()[0]
    tickers_with_calls = conn.execute(
        "SELECT COUNT(DISTINCT symbol) FROM options_history WHERE side='call'").fetchone()[0]
    call_errors = conn.execute('SELECT COUNT(*) FROM fetch_errors_calls').fetchone()[0] \
        if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fetch_errors_calls'").fetchone() else 0
    size_mb = os.path.getsize(DB_PATH) / 1024 / 1024
    print(f"📊 Options History Database (CALLS view)")
    print(f"   Call records:   {call_records:,}")
    print(f"   Put records:    {put_records:,} (existing)")
    print(f"   Tickers w/calls: {tickers_with_calls}")
    print(f"   Call errors:    {call_errors:,}")
    print(f"   DB size:        {size_mb:.1f} MB")
    if call_records:
        oldest = conn.execute("SELECT MIN(trade_date) FROM options_history WHERE side='call'").fetchone()[0]
        newest = conn.execute("SELECT MAX(trade_date) FROM options_history WHERE side='call'").fetchone()[0]
        print(f"   Range:          {oldest} → {newest}")
    conn.close()


if __name__ == '__main__':
    # Hardcoded: AAPL, 5 years. Resume logic will skip already-downloaded dates.
    download(years=5, ticker_filter='AAPL')