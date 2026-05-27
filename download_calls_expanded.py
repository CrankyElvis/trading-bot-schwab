"""
download_calls_expanded.py
─────────────────────────────────────────────────────────────────────────────
Downloads call option history for the 25 Tier 1 + Tier 2 expansion tickers
identified by analyze_score_coverage.py.

Identical logic to download_calls_universe.py, only the UNIVERSE list differs
and the error table is renamed so resume logic doesn't collide.

Usage on server:
  cd /root/trading-bot
  nohup python3 -u download_calls_expanded.py > /tmp/calls_expanded.log 2>&1 &
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
YEARS          = 5  # 5yr request, but expect ~1yr returned for most tickers

# Tier 1 (10): highest very-high-score days in the score CSV
# Tier 2 (15): strong supporting tech mid-large caps
UNIVERSE = [
    # Tier 1
    'APP', 'SMCI', 'MSTR', 'PLTR', 'DDOG', 'NET', 'AFRM', 'DASH', 'SNOW', 'CRWD',
    # Tier 2
    'AVGO', 'TSM', 'MRVL', 'MU', 'KLAC', 'LRCX', 'IBKR', 'SMH', 'ASML', 'NFLX',
    'ZS', 'PANW', 'AMAT', 'MEDP', 'INSM',
]


# ── Database ──────────────────────────────────────────────────────────────

def init_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
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
        CREATE TABLE IF NOT EXISTS fetch_errors_calls_expanded (
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
        'SELECT 1 FROM fetch_errors_calls_expanded WHERE symbol=? AND trade_date=? LIMIT 1',
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
        INSERT OR REPLACE INTO fetch_errors_calls_expanded
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

    print(f"[*] EXPANDED CALL Downloader")
    print(f"   Tickers: {len(UNIVERSE)} | Days: {len(trade_days)} | Total: {total:,}")
    print(f"   Side:    {SIDE} ({OTM_PCT*100:.0f}% OTM, ~{TARGET_DTE} DTE)")
    print(f"   DB:      {DB_PATH}")
    print()

    t0 = time.time()

    for ticker_idx, ticker in enumerate(UNIVERSE, 1):
        print(f"\n[{ticker_idx}/{len(UNIVERSE)}] -- {ticker} --")
        t_fetched = t_skipped = t_errors = 0
        t_start = time.time()

        for trade_date in trade_days:
            done += 1
            if already_fetched(conn, ticker, trade_date):
                skipped += 1
                t_skipped += 1
                continue

            try:
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

                elapsed = time.time() - t0
                rate = fetched / max(elapsed, 1)
                eta_h = (total - done - skipped) / max(rate, 0.01) / 3600
                print(f"   {ticker}: {trade_date} mid=${mid:.2f} "
                      f"({t_fetched} done, credits={credits_used}, ETA={eta_h:.1f}h)",
                      end='\r')

            except KeyboardInterrupt:
                print(f"\n\n[!] Interrupted -- progress saved. Re-run to continue.")
                conn.close()
                sys.exit(0)
            except Exception as e:
                save_error(conn, ticker, trade_date, str(e))
                errors += 1; t_errors += 1

        t_elapsed = (time.time() - t_start) / 60
        print(f"\n   {ticker}: fetched={t_fetched} skipped={t_skipped} "
              f"errors={t_errors} ({t_elapsed:.1f} min)")

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"[+] Expanded call download complete")
    print(f"   Fetched:  {fetched:,} records")
    print(f"   Skipped:  {skipped:,} (already in DB)")
    print(f"   Errors:   {errors:,}")
    print(f"   Credits:  {credits_used:,}")
    print(f"   Time:     {elapsed:.1f} min")
    print(f"   DB:       {DB_PATH}")
    conn.close()


if __name__ == '__main__':
    download()