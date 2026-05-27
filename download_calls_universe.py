"""
download_calls_universe.py
──────────────────────────────────────────────────────────────────────────────
Downloads call option history for the 22 tickers in the DB that already have
puts but no calls. Same params as download_calls_history.py (35-45 DTE, 8% OTM)
but iterates over the multi-ticker universe instead of being hardcoded to AAPL.

Universe matched against tickers already in DB with puts but no calls:
  AMZN, BAC, DIA, GDX, GOOGL, GS, IWM, JPM, META, MSFT, NVDA,
  QQQ, SCHP, SPY, TSLA, VIXY, VTIP, XLE, XLF, XLI, XLK, XLV

Expected results per ticker (based on existing 1yr put data):
  - ~220 call records
  - ~2,200 credits per ticker
  - ~3-6 min per ticker
  Total: ~4,800 records, ~12-15k credits, ~75 min

Resume logic: skips (ticker, date) pairs that already have a call record OR a
call-side error. Safe to re-run.

Usage:
  python download_calls_universe.py
"""

import os
import sys
import time
import sqlite3
import requests
from datetime import datetime, timedelta, date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────

DB_PATH        = '/root/trading-bot/data/options_history.db'
API_BASE       = 'https://api.marketdata.app/v1'
TARGET_DTE     = 45
OTM_PCT        = 0.08
RATE_LIMIT_RPS = 5
SLEEP_BETWEEN  = 1.0 / RATE_LIMIT_RPS
SIDE           = 'call'
YEARS          = 1  # Only ~1yr of history actually available for non-AAPL tickers

# The 22 tickers that need calls
UNIVERSE = [
    'AMZN', 'BAC', 'DIA', 'GDX', 'GOOGL', 'GS', 'IWM', 'JPM',
    'META', 'MSFT', 'NVDA', 'QQQ', 'SCHP', 'SPY', 'TSLA', 'VIXY',
    'VTIP', 'XLE', 'XLF', 'XLI', 'XLK', 'XLV',
]


# ── Database ──────────────────────────────────────────────────────────────

def init_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)  # 30s wait on locks
    conn.execute('PRAGMA journal_mode=WAL')        # allow concurrent reads while writing
    conn.execute('PRAGMA busy_timeout=30000')      # 30 seconds before raising locked error
    conn.execute('PRAGMA synchronous=NORMAL')      # faster writes, still safe
    conn.execute('''
        CREATE TABLE IF NOT EXISTS options_history (
            symbol TEXT NOT NULL, trade_date TEXT NOT NULL,
            expiration TEXT NOT NULL, strike REAL NOT NULL,
            side TEXT NOT NULL, dte INTEGER, bid REAL, ask REAL, mid REAL,
            volume INTEGER, open_interest INTEGER, underlying_price REAL,
            in_the_money INTEGER, fetched_at TEXT,
            PRIMARY KEY (symbol, trade_date, expiration, strike, side)
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS fetch_errors_calls (
            symbol TEXT, trade_date TEXT, error TEXT, fetched_at TEXT,
            PRIMARY KEY (symbol, trade_date)
        )
    ''')
    conn.commit()
    return conn


def already_fetched(conn, symbol, trade_date):
    cur = conn.execute(
        "SELECT 1 FROM options_history WHERE symbol=? AND trade_date=? AND side='call' LIMIT 1",
        (symbol, trade_date))
    if cur.fetchone():
        return True
    cur = conn.execute(
        'SELECT 1 FROM fetch_errors_calls WHERE symbol=? AND trade_date=? LIMIT 1',
        (symbol, trade_date))
    return cur.fetchone() is not None


def save_record(conn, record):
    conn.execute('''
        INSERT OR REPLACE INTO options_history
        (symbol, trade_date, expiration, strike, side, dte, bid, ask, mid,
         volume, open_interest, underlying_price, in_the_money, fetched_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', record)
    conn.commit()


def save_error(conn, symbol, trade_date, error):
    conn.execute('''
        INSERT OR REPLACE INTO fetch_errors_calls
        (symbol, trade_date, error, fetched_at) VALUES (?,?,?,?)
    ''', (symbol, trade_date, str(error), datetime.now().isoformat()))
    conn.commit()


# ── API ───────────────────────────────────────────────────────────────────

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
    return r.json().get('expirations', [])


def find_target_expiration(expirations, trade_date_str, target_dte=45):
    trade_dt = datetime.strptime(trade_date_str, '%Y-%m-%d').date()
    best_exp, best_diff = None, 9999
    for exp in expirations:
        try:
            exp_dt = datetime.strptime(exp, '%Y-%m-%d').date()
            dte = (exp_dt - trade_dt).days
            if dte < 21:
                continue
            diff = abs(dte - target_dte)
            if diff < best_diff:
                best_diff, best_exp = diff, exp
        except Exception:
            continue
    return best_exp


def get_trading_days(years=5):
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=years * 365)
    holidays = set()
    for yr in range(start.year, end.year + 1):
        holidays.update([date(yr, 1, 1), date(yr, 7, 4), date(yr, 12, 25)])
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5 and d not in holidays:
            days.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    return days


# ── Main ──────────────────────────────────────────────────────────────────

def download():
    conn = init_db(DB_PATH)
    if not os.environ.get('MARKETDATA_API_KEY'):
        print("[X] MARKETDATA_API_KEY not set")
        sys.exit(1)

    trade_days = get_trading_days(YEARS)
    total = len(UNIVERSE) * len(trade_days)
    done = skipped = errors = fetched = credits_used = 0
    MAX_RUNTIME_SEC = 8 * 3600  # 8 hour kill switch

    print(f"[*] CALL Universe Downloader")
    print(f"   Tickers:     {len(UNIVERSE)} | Days: {len(trade_days)} | Total attempts: {total:,}")
    print(f"   Side:        {SIDE} ({OTM_PCT*100:.0f}% OTM, ~{TARGET_DTE} DTE)")
    print(f"   DB:          {DB_PATH}")
    print(f"   Years:       {YEARS}  (data limit prevents 5yr depth on non-AAPL tickers)")
    print(f"   Max runtime: {MAX_RUNTIME_SEC/3600:.0f} hours (auto-stop)")
    print(f"   Estimate:    ~{len(UNIVERSE) * len(trade_days) * 0.4 / 60:.0f} min if all dates fetch",
          flush=True)
    print()

    t0 = time.time()

    for ticker_idx, ticker in enumerate(UNIVERSE, 1):
        elapsed_min = (time.time() - t0) / 60
        print(f"\n[{ticker_idx}/{len(UNIVERSE)}] -- {ticker} -- "
              f"(elapsed: {elapsed_min:.1f} min)", flush=True)
        t_fetched = t_skipped = t_errors = 0
        t_start = time.time()

        for trade_date in trade_days:
            # Runtime kill switch
            if time.time() - t0 > MAX_RUNTIME_SEC:
                print(f"\n\n[!] Max runtime ({MAX_RUNTIME_SEC/3600:.0f}h) reached -- stopping safely.",
                      flush=True)
                conn.close()
                sys.exit(0)

            done += 1
            if already_fetched(conn, ticker, trade_date):
                skipped += 1
                t_skipped += 1
                continue

            try:
                # Step 1: expirations
                time.sleep(SLEEP_BETWEEN)
                expirations = get_expirations(ticker, trade_date)
                credits_used += 1
                if not expirations:
                    save_error(conn, ticker, trade_date, 'no_expirations')
                    errors += 1; t_errors += 1
                    continue

                expiration = find_target_expiration(expirations, trade_date, TARGET_DTE)
                if not expiration:
                    save_error(conn, ticker, trade_date, 'no_valid_expiration')
                    errors += 1; t_errors += 1
                    continue

                # Step 2: chain for the call side
                time.sleep(SLEEP_BETWEEN)
                url = f'{API_BASE}/options/chain/{ticker}/'
                r = requests.get(url, headers=get_headers(),
                                 params={'date': trade_date, 'expiration': expiration,
                                         'side': SIDE, 'minOpenInterest': 10},
                                 timeout=15)
                credits_used += 1
                if r.status_code not in (200, 203):
                    save_error(conn, ticker, trade_date, f'chain_error_{r.status_code}')
                    errors += 1; t_errors += 1
                    continue

                chain = r.json()
                if chain.get('s') != 'ok':
                    save_error(conn, ticker, trade_date, 'chain_no_data')
                    errors += 1; t_errors += 1
                    continue

                under_prices = chain.get('underlyingPrice', [])
                if not under_prices:
                    save_error(conn, ticker, trade_date, 'no_underlying_price')
                    errors += 1; t_errors += 1
                    continue

                underlying = under_prices[0]
                target_strike = round(underlying * (1 + OTM_PCT), 0)

                strikes = chain.get('strike', [])
                if not strikes:
                    save_error(conn, ticker, trade_date, 'no_strikes')
                    errors += 1; t_errors += 1
                    continue

                nearest_strike = float(min(strikes, key=lambda s: abs(float(s) - target_strike)))

                # Extract data at nearest strike index
                try:
                    idx = [float(s) for s in strikes].index(nearest_strike)
                except ValueError:
                    diffs = [abs(float(s) - nearest_strike) for s in strikes]
                    idx = diffs.index(min(diffs))

                def safe_get(lst, i):
                    try: return lst[i]
                    except: return None

                bid = safe_get(chain.get('bid', []), idx)
                ask = safe_get(chain.get('ask', []), idx)
                mid = safe_get(chain.get('mid', []), idx)
                dte = safe_get(chain.get('dte', []), idx)
                vol = safe_get(chain.get('volume', []), idx)
                oi = safe_get(chain.get('openInterest', []), idx)
                itm = 1 if safe_get(chain.get('inTheMoney', []), idx) else 0

                if mid is None:
                    save_error(conn, ticker, trade_date, 'no_mid_price')
                    errors += 1; t_errors += 1
                    continue

                record = (ticker, trade_date, expiration, nearest_strike, SIDE,
                          dte, bid, ask, mid, vol, oi, underlying, itm,
                          datetime.now().isoformat())
                save_record(conn, record)
                fetched += 1; t_fetched += 1
                credits_used += 1

                # Progress line (newline-mode for log file compatibility)
                elapsed = time.time() - t0
                rate = fetched / max(elapsed, 1)
                eta_h = (total - done - skipped) / max(rate, 0.01) / 3600
                # Only print every 20 records to keep logs readable
                if t_fetched % 20 == 0:
                    print(f"   {ticker}: {trade_date} mid=${mid:.2f} "
                          f"({t_fetched} done, credits={credits_used}, ETA={eta_h:.1f}h)",
                          flush=True)

            except KeyboardInterrupt:
                print(f"\n\n[!] Interrupted -- progress saved. Re-run to continue.")
                conn.close()
                sys.exit(0)
            except Exception as e:
                save_error(conn, ticker, trade_date, str(e))
                errors += 1; t_errors += 1

        t_elapsed = (time.time() - t_start) / 60
        print(f"\n   {ticker}: fetched={t_fetched} skipped={t_skipped} "
              f"errors={t_errors} ({t_elapsed:.1f} min)", flush=True)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"[+] Universe call download complete")
    print(f"   Fetched:  {fetched:,} records")
    print(f"   Skipped:  {skipped:,} (already in DB)")
    print(f"   Errors:   {errors:,}")
    print(f"   Credits:  {credits_used:,}")
    print(f"   Time:     {elapsed:.1f} min")
    print(f"   DB:       {DB_PATH}")
    conn.close()


if __name__ == '__main__':
    download()
