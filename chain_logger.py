"""
chain_logger.py
─────────────────────────────────────────────────────────────────────────────
Fetches live options chains from Schwab and persists them to chains_live.db.

Used by options_manager.py:
  - Replaces Black-Scholes premium estimates with real bid/ask/mid
  - Replaces estimate_iv(vixy) with per-strike IV from the chain
  - Captures greeks (delta, gamma, theta, vega, rho) for later analysis

Every cycle that evaluates a CSP or long call calls fetch_and_log_chain().
That means we get a snapshot of the real options market for every candidate
ticker the bot ever considers, whether or not it ends up trading.

After 3-6 months of paper trading, this DB becomes the basis for:
  - Real delta-based strike selection
  - Actual IV-rank computation (vs UW's API which can be stale or limited)
  - Spread strategy backtesting (multi-strike data finally available)
  - Validation that the bot's premium estimates were anywhere close to reality

Schema:
  chain_fetches      one row per (timestamp, symbol) -- when we queried Schwab
  chain_quotes       one row per strike in the chain -- the actual market data

Design:
  - Best-effort: never raises, never breaks the live bot
  - WAL mode for concurrent reads
  - Top-20 strikes by abs(strike - underlying) per side -- not the whole chain
  - 5-minute in-memory cache to avoid hammering Schwab on the same symbol
"""

import os
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = '/root/trading-bot/data/chains_live.db'
DB_PATH = os.environ.get('CHAIN_LOG_DB', DEFAULT_DB_PATH)

# How many strikes to keep per (side) -- closest to underlying.
# 20 captures the realistic trading range (8-15% OTM on both sides + a few ATM).
TOP_N_STRIKES_PER_SIDE = 20

# How long to cache a chain in memory before re-querying Schwab for the same symbol.
# Saves API hits when multiple strategies evaluate the same ticker in one cycle.
CACHE_TTL_SECONDS = 300   # 5 minutes

# Thread safety
_lock = threading.Lock()
_init_done = False

# In-process cache: {symbol: (fetched_at_ts, chain_dict)}
_chain_cache: dict = {}


# ── Schema initialization ─────────────────────────────────────────────────────

def _init_db():
    """Create tables if they don't exist. Idempotent. Best-effort."""
    global _init_done
    if _init_done:
        return
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")

        # One row per (timestamp, symbol) query
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chain_fetches (
                fetch_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                fetched_at      TEXT NOT NULL,
                symbol          TEXT NOT NULL,
                underlying_price REAL,
                source          TEXT,     -- 'schwab' / 'cache' / 'error'
                n_strikes       INTEGER,
                requested_by    TEXT,     -- which caller (evaluate_csp, evaluate_long_call, etc.)
                error_msg       TEXT,
                raw_keys        TEXT      -- JSON keys present in response (for debugging schema drift)
            )
        ''')

        # One row per strike in the chain
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chain_quotes (
                quote_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                fetch_id        INTEGER NOT NULL,
                fetched_at      TEXT NOT NULL,
                symbol          TEXT NOT NULL,
                side            TEXT NOT NULL,        -- 'call' / 'put'
                strike          REAL NOT NULL,
                expiration      TEXT NOT NULL,
                dte             INTEGER,

                -- Pricing
                bid             REAL,
                ask             REAL,
                mark            REAL,
                last            REAL,
                mid             REAL,                 -- (bid + ask) / 2

                -- Volume / OI
                volume          INTEGER,
                open_interest   INTEGER,

                -- Greeks (the whole reason this module exists)
                delta           REAL,
                gamma           REAL,
                theta           REAL,
                vega            REAL,
                rho             REAL,

                -- Implied volatility
                iv              REAL,                 -- volatility from chain
                time_value      REAL,
                intrinsic_value REAL,

                -- Underlying snapshot at time of fetch
                underlying_price REAL,

                FOREIGN KEY (fetch_id) REFERENCES chain_fetches(fetch_id)
            )
        ''')

        # Indexes for query performance
        conn.execute('CREATE INDEX IF NOT EXISTS idx_quotes_symbol_time ON chain_quotes(symbol, fetched_at)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_quotes_fetch ON chain_quotes(fetch_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_fetches_symbol_time ON chain_fetches(symbol, fetched_at)')

        conn.commit()
        conn.close()
        _init_done = True
    except Exception as e:
        print(f"  [chain_logger] WARNING: DB init failed: {e}")


# ── Schwab chain parsing ──────────────────────────────────────────────────────

def _safe(d, key, default=None):
    """Best-effort field extraction from Schwab's chain response."""
    try:
        v = d.get(key, default)
        if v is None or v == 'NaN':
            return default
        return v
    except Exception:
        return default


def _parse_schwab_chain(chain_json: dict, fetched_at: str) -> dict:
    """
    Parses Schwab option chain JSON into a flat list of strike records.

    Schwab response shape:
      {
        'symbol': 'AAPL',
        'status': 'SUCCESS',
        'underlyingPrice': 195.50,
        'underlying': {...},
        'callExpDateMap': {
          '2026-06-19:25': {     # 'YYYY-MM-DD:DTE'
            '200.0': [           # strike
              {  # this dict is one option contract
                'putCall': 'CALL',
                'symbol': 'AAPL  260619C00200000',
                'strikePrice': 200.0,
                'bid': 2.5, 'ask': 2.6, 'last': 2.55, 'mark': 2.55,
                'totalVolume': 1500, 'openInterest': 12000,
                'volatility': 24.5,
                'delta': 0.45, 'gamma': 0.03, 'theta': -0.08, 'vega': 0.15, 'rho': 0.05,
                'timeValue': 2.55, 'intrinsicValue': 0.0,
                'daysToExpiration': 25,
                'expirationDate': '2026-06-19T20:00:00.000+00:00',
                ...
              }
            ],
            ...
          },
          ...
        },
        'putExpDateMap': { ... same shape ... }
      }

    Returns: {'underlying_price': float, 'strikes': [strike_records...], 'raw_keys': [..]}
    """
    if not chain_json or not isinstance(chain_json, dict):
        return {'underlying_price': None, 'strikes': [], 'raw_keys': []}

    underlying = _safe(chain_json, 'underlyingPrice')
    if underlying is None:
        underlying = _safe(_safe(chain_json, 'underlying', {}) or {}, 'last')

    raw_keys = list(chain_json.keys())
    strikes_out = []

    for side, map_key in [('call', 'callExpDateMap'), ('put', 'putExpDateMap')]:
        exp_map = chain_json.get(map_key, {}) or {}
        if not isinstance(exp_map, dict):
            continue

        for exp_key, strikes_dict in exp_map.items():
            # exp_key looks like '2026-06-19:25' -- the part before colon is the date
            expiration = exp_key.split(':')[0] if ':' in exp_key else exp_key

            if not isinstance(strikes_dict, dict):
                continue

            for strike_str, contracts in strikes_dict.items():
                if not isinstance(contracts, list) or not contracts:
                    continue
                c = contracts[0]   # Schwab returns a list of length 1 per strike
                if not isinstance(c, dict):
                    continue

                try:
                    strike = float(strike_str)
                except (ValueError, TypeError):
                    continue

                bid = _safe(c, 'bid')
                ask = _safe(c, 'ask')
                mid = None
                try:
                    if bid is not None and ask is not None:
                        mid = round((float(bid) + float(ask)) / 2, 4)
                except (TypeError, ValueError):
                    mid = None

                strikes_out.append({
                    'side':            side,
                    'strike':          strike,
                    'expiration':      expiration,
                    'dte':             _safe(c, 'daysToExpiration'),
                    'bid':             bid,
                    'ask':             ask,
                    'mark':            _safe(c, 'mark'),
                    'last':            _safe(c, 'last'),
                    'mid':             mid,
                    'volume':          _safe(c, 'totalVolume'),
                    'open_interest':   _safe(c, 'openInterest'),
                    'delta':           _safe(c, 'delta'),
                    'gamma':           _safe(c, 'gamma'),
                    'theta':           _safe(c, 'theta'),
                    'vega':            _safe(c, 'vega'),
                    'rho':             _safe(c, 'rho'),
                    'iv':              _safe(c, 'volatility'),
                    'time_value':      _safe(c, 'timeValue'),
                    'intrinsic_value': _safe(c, 'intrinsicValue'),
                })

    return {
        'underlying_price': underlying,
        'strikes':          strikes_out,
        'raw_keys':         raw_keys,
    }


def _filter_top_n(strikes: list, underlying: float, n_per_side: int) -> list:
    """Keep only the N strikes closest to underlying per side (call/put)."""
    if not strikes or underlying is None:
        return strikes

    by_side = {'call': [], 'put': []}
    for s in strikes:
        side = s.get('side')
        if side in by_side:
            by_side[side].append(s)

    out = []
    for side, lst in by_side.items():
        # Sort by distance to underlying, keep top N
        lst.sort(key=lambda x: abs(float(x['strike']) - float(underlying)))
        out.extend(lst[:n_per_side])

    return out


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_and_log_chain(client, symbol: str, requested_by: str = 'unknown',
                         dte_min: int = 14, dte_max: int = 60) -> dict | None:
    """
    Fetch Schwab option chain for `symbol`, log everything to chains_live.db,
    and return the parsed chain for the caller to use.

    Args:
        client:        Authenticated schwab-py client (from auth.authenticate())
        symbol:        Ticker symbol
        requested_by:  Name of caller (for diagnostics in chain_fetches table)
        dte_min:       Minimum days to expiration to include
        dte_max:       Maximum days to expiration to include

    Returns:
        dict with shape:
          {
            'symbol': str,
            'underlying_price': float,
            'strikes': [list of strike dicts with side/strike/expiration/bid/ask/delta/iv/...],
            'source': 'schwab' | 'cache' | 'error',
          }
        Returns None only on catastrophic failure.

    Never raises -- failure is logged to chain_fetches.error_msg.
    """
    _init_db()

    # Check in-process cache first
    cached = _chain_cache.get(symbol)
    now_ts = time.time()
    if cached and (now_ts - cached[0]) < CACHE_TTL_SECONDS:
        return cached[1]

    fetched_at = datetime.now().isoformat()
    chain_data = None
    error_msg = None
    raw_keys = []

    # Fetch from Schwab
    try:
        # schwab-py client.get_option_chain signature varies by version.
        # We try a few common shapes:
        chain_resp = None
        try:
            # Most common: keyword args with date range
            from_date = (datetime.now() + timedelta(days=dte_min)).date()
            to_date   = (datetime.now() + timedelta(days=dte_max)).date()
            chain_resp = client.get_option_chain(
                symbol,
                from_date=from_date,
                to_date=to_date,
                include_underlying_quote=True,
            )
        except (TypeError, AttributeError):
            # Fallback: positional / no date filter
            try:
                chain_resp = client.get_option_chain(symbol)
            except Exception as e2:
                error_msg = f"get_option_chain failed: {e2}"

        if chain_resp is not None:
            chain_json = chain_resp.json() if hasattr(chain_resp, 'json') else chain_resp
            parsed = _parse_schwab_chain(chain_json, fetched_at)
            raw_keys = parsed['raw_keys']
            underlying = parsed['underlying_price']
            all_strikes = parsed['strikes']

            # Filter DTE if Schwab didn't already
            if dte_min or dte_max:
                all_strikes = [
                    s for s in all_strikes
                    if (s.get('dte') is None) or (dte_min <= int(s['dte']) <= dte_max)
                ]

            # Keep top N nearest underlying per side
            top_strikes = _filter_top_n(all_strikes, underlying, TOP_N_STRIKES_PER_SIDE)

            chain_data = {
                'symbol':           symbol,
                'underlying_price': underlying,
                'strikes':          top_strikes,
                'source':           'schwab',
                'fetched_at':       fetched_at,
            }
    except Exception as e:
        error_msg = f"chain fetch exception: {e}"

    # Even on error, log the attempt so we can debug
    if chain_data is None:
        chain_data = {
            'symbol':           symbol,
            'underlying_price': None,
            'strikes':          [],
            'source':           'error',
            'fetched_at':       fetched_at,
        }

    # Persist
    try:
        with _lock:
            conn = sqlite3.connect(DB_PATH, timeout=10)
            try:
                cur = conn.execute('''
                    INSERT INTO chain_fetches
                    (fetched_at, symbol, underlying_price, source, n_strikes,
                     requested_by, error_msg, raw_keys)
                    VALUES (?,?,?,?,?,?,?,?)
                ''', (
                    fetched_at, symbol, chain_data['underlying_price'],
                    chain_data['source'], len(chain_data['strikes']),
                    requested_by, error_msg,
                    json.dumps(raw_keys) if raw_keys else None,
                ))
                fetch_id = cur.lastrowid

                rows = []
                for s in chain_data['strikes']:
                    rows.append((
                        fetch_id, fetched_at, symbol,
                        s.get('side'), s.get('strike'), s.get('expiration'),
                        s.get('dte'),
                        s.get('bid'), s.get('ask'), s.get('mark'), s.get('last'), s.get('mid'),
                        s.get('volume'), s.get('open_interest'),
                        s.get('delta'), s.get('gamma'), s.get('theta'),
                        s.get('vega'), s.get('rho'),
                        s.get('iv'), s.get('time_value'), s.get('intrinsic_value'),
                        chain_data['underlying_price'],
                    ))
                if rows:
                    conn.executemany('''
                        INSERT INTO chain_quotes
                        (fetch_id, fetched_at, symbol, side, strike, expiration, dte,
                         bid, ask, mark, last, mid, volume, open_interest,
                         delta, gamma, theta, vega, rho,
                         iv, time_value, intrinsic_value, underlying_price)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ''', rows)
                conn.commit()
            finally:
                conn.close()

        n = len(chain_data['strikes'])
        if chain_data['source'] == 'schwab' and n > 0:
            print(f"  [chain_logger] {symbol}: logged {n} strikes "
                  f"(underlying ${chain_data['underlying_price']})")
        elif error_msg:
            print(f"  [chain_logger] {symbol}: FAILED -- {error_msg}")
    except Exception as e:
        print(f"  [chain_logger] WARNING: persist failed for {symbol}: {e}")

    # Cache successful fetches only
    if chain_data['source'] == 'schwab':
        _chain_cache[symbol] = (now_ts, chain_data)

    return chain_data


# ── Convenience helpers for options_manager ──────────────────────────────────

def find_strike(chain: dict, side: str, target_strike: float = None,
                 target_delta: float = None, target_otm_pct: float = None) -> dict | None:
    """
    Pick the strike from a chain that best matches the target.

    Priority:
      1. If target_delta given: find strike with delta closest to target
      2. If target_otm_pct given: find strike at underlying * (1 ± otm_pct)
      3. If target_strike given: find strike closest to that price

    Returns the strike dict or None.
    """
    if not chain or not chain.get('strikes'):
        return None

    candidates = [s for s in chain['strikes'] if s.get('side') == side]
    if not candidates:
        return None

    underlying = chain.get('underlying_price') or 0

    if target_delta is not None:
        # For calls: delta is 0 to 1. For puts: -1 to 0. Use abs() for matching.
        valid = [c for c in candidates if c.get('delta') is not None]
        if valid:
            return min(valid, key=lambda c: abs(abs(float(c['delta'])) - abs(target_delta)))

    if target_otm_pct is not None and underlying > 0:
        if side == 'call':
            target = underlying * (1 + target_otm_pct)
        else:
            target = underlying * (1 - target_otm_pct)
        return min(candidates, key=lambda c: abs(float(c['strike']) - target))

    if target_strike is not None:
        return min(candidates, key=lambda c: abs(float(c['strike']) - target_strike))

    # Default: ATM
    if underlying > 0:
        return min(candidates, key=lambda c: abs(float(c['strike']) - underlying))

    return None


# ── Read-side helpers ─────────────────────────────────────────────────────────

def db_path() -> str:
    return DB_PATH


def summary_stats():
    """Print a quick summary of logged data."""
    if not os.path.exists(DB_PATH):
        print(f"No DB at {DB_PATH} yet -- nothing logged.")
        return
    conn = sqlite3.connect(DB_PATH)
    fetches    = conn.execute("SELECT COUNT(*) FROM chain_fetches").fetchone()[0]
    successes  = conn.execute("SELECT COUNT(*) FROM chain_fetches WHERE source='schwab'").fetchone()[0]
    quotes     = conn.execute("SELECT COUNT(*) FROM chain_quotes").fetchone()[0]
    symbols    = conn.execute("SELECT COUNT(DISTINCT symbol) FROM chain_quotes").fetchone()[0]
    with_greeks = conn.execute("SELECT COUNT(*) FROM chain_quotes WHERE delta IS NOT NULL").fetchone()[0]
    first      = conn.execute("SELECT MIN(fetched_at) FROM chain_fetches").fetchone()[0]
    last       = conn.execute("SELECT MAX(fetched_at) FROM chain_fetches").fetchone()[0]
    conn.close()
    print(f"chain_logger summary:")
    print(f"  DB:                 {DB_PATH}")
    print(f"  Total fetches:      {fetches}")
    print(f"  Successful:         {successes}  ({100*successes/max(fetches,1):.1f}%)")
    print(f"  Total strike quotes: {quotes}")
    print(f"  With greeks (delta): {with_greeks}  ({100*with_greeks/max(quotes,1):.1f}%)")
    print(f"  Unique symbols:     {symbols}")
    print(f"  First fetch:        {first}")
    print(f"  Last fetch:         {last}")


if __name__ == '__main__':
    summary_stats()