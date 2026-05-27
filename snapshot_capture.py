"""
snapshot_capture.py
─────────────────────────────────────────────────────────────────────────────
Persists every trading cycle's full decision context to snapshots.db.

Captures the live-only data that we can NEVER reconstruct from historical APIs:
  - Real-time UW options flow at the moment of decision (per-symbol)
  - Dark pool prints with timestamps (per-symbol)
  - Live bid/ask spreads
  - Fear & Greed Index intraday reading
  - VIX term structure at decision time
  - Full regime state (regime + position size + macro override status)
  - Per-ticker bot decisions: scored? blocked? traded? why?
  - Source tagging (DEFAULT_UNIVERSE vs politician/wsb/earnings injection)

Every row foreign-keys to bot_versions.version_id, so analysis can answer:
"did the X signal start contributing edge after code change Y on date Z?"

Design choices:
  - Single SQLite DB at /root/trading-bot/data/snapshots.db (same DB as
    bot_version so foreign keys are real, not implicit)
  - One row per (cycle_id, ticker) in decisions table — analytics is fast
  - JSON blobs for nested structures (quotes, uw_flow, signals dict)
  - Best-effort: every function wrapped in try/except, never crashes the bot
  - WAL mode for concurrent reads while bot is writing

Usage from main.py:
    from snapshot_capture import archive_snapshot, archive_decisions

    # After collect_snapshot + regime evaluation, before scoring:
    cycle_id = archive_snapshot(
        snapshot=snapshot, regime_state=regime_state,
        term_structure=term_structure, macro_state=macro,
        cycle_name=cycle_name, version_id=version_id,
        injected_symbols=injected_symbols,
    )

    # After scoring + blocker checks + trade decisions:
    archive_decisions(
        cycle_id=cycle_id, version_id=version_id,
        scored_results=score_result.all_scores,
        passed_risk=passed_symbols,
        blocked={sym: blockers for sym, blockers in blocked_dict.items()},
        traded=trade_results,
    )
"""

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional, Any


# ── Configuration ─────────────────────────────────────────────────────────────

DB_PATH = os.environ.get('SNAPSHOTS_DB', '/root/trading-bot/data/snapshots.db')

_lock = threading.Lock()
_init_done = False


# ── Schema ────────────────────────────────────────────────────────────────────

def _init_db():
    """Create tables if they don't exist. Idempotent."""
    global _init_done
    if _init_done:
        return
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")

        # NOTE: bot_versions table is created by bot_version.py — don't recreate
        # here. We just reference it. If bot_version hasn't initialized yet,
        # foreign key will be NULL (we allow that).

        # One row per scoring cycle. Captures market-level state + bot state.
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cycle_snapshots (
                cycle_id            INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id          INTEGER,
                cycle_time          TEXT NOT NULL,
                cycle_name          TEXT,
                trading_date        TEXT NOT NULL,

                -- Market state at decision time
                vixy                REAL,
                vixy_30d_avg        REAL,
                vixy_30d_std        REAL,
                vix_zscore          REAL,

                -- VIX term structure
                vix9d               REAL,
                vix_30d             REAL,
                vix3m               REAL,
                term_signal         TEXT,
                halt_entries        INTEGER,

                -- Regime
                regime              TEXT,
                previous_regime     TEXT,
                position_size       REAL,
                in_pause            INTEGER,
                mean_reversion_signal INTEGER,

                -- Macro sentinel
                macro_score         REAL,
                macro_warning       TEXT,
                macro_override      INTEGER,

                -- Sentiment
                fear_greed_score    INTEGER,
                fear_greed_label    TEXT,

                -- Universe context
                universe_size       INTEGER,
                injected_count      INTEGER,
                injected_json       TEXT,

                -- Full payload blobs — JSON for queryability, gzip later if size matters
                quotes_json         TEXT,
                uw_darkpool_json    TEXT,

                created_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Per-symbol UW flow at decision time. Split out because flow per symbol
        # can have many rows — keeping it normalized lets us query "all sweeps
        # above $50k on AAPL in May 2026" trivially.
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cycle_uw_flow (
                flow_id             INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id            INTEGER NOT NULL,
                version_id          INTEGER,
                symbol              TEXT NOT NULL,
                flow_json           TEXT NOT NULL,
                row_count           INTEGER,
                FOREIGN KEY (cycle_id) REFERENCES cycle_snapshots(cycle_id)
            )
        ''')

        # Per-ticker decision: what did the bot conclude for this symbol this
        # cycle? This is the row that gets joined against actual fills for
        # reconciliation later.
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cycle_decisions (
                decision_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id            INTEGER NOT NULL,
                version_id          INTEGER,
                cycle_time          TEXT NOT NULL,
                trading_date        TEXT NOT NULL,
                symbol              TEXT NOT NULL,

                -- Source: 'universe' | 'politician' | 'wsb' | 'earnings_surprise'
                source              TEXT,

                -- Risk gate
                passed_risk         INTEGER,
                blockers            TEXT,             -- comma-separated blocker reasons

                -- Scoring (may be NULL if blocked before scoring)
                total_score         REAL,
                qualifies           INTEGER,
                direction           TEXT,
                signals_json        TEXT,             -- full per-signal scores
                weighted_json       TEXT,             -- weighted contributions

                -- Decision
                was_candidate       INTEGER,          -- selected for trade attempt
                attempted_trade     INTEGER,          -- main.py tried to execute
                trade_succeeded     INTEGER,          -- paper_buy returned True
                trade_shares        INTEGER,
                trade_price         REAL,
                trade_reason_skip   TEXT,             -- e.g. 'bearish_signal_non_crisis', 'price_unavailable'

                FOREIGN KEY (cycle_id) REFERENCES cycle_snapshots(cycle_id)
            )
        ''')

        # Indexes for the analysis queries that will matter most
        conn.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_date '
                     'ON cycle_snapshots(trading_date)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_version '
                     'ON cycle_snapshots(version_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_decisions_symbol_date '
                     'ON cycle_decisions(symbol, trading_date)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_decisions_cycle '
                     'ON cycle_decisions(cycle_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_decisions_version '
                     'ON cycle_decisions(version_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_uw_flow_cycle '
                     'ON cycle_uw_flow(cycle_id, symbol)')

        conn.commit()
        conn.close()
        _init_done = True
    except Exception as e:
        print(f"  [snapshot_capture] WARNING: DB init failed: {e}")


# ── Serialization helpers ─────────────────────────────────────────────────────

def _df_to_json(df) -> Optional[str]:
    """Convert a pandas DataFrame to a JSON list-of-records. Returns None on
    empty / None / failure. Always best-effort, never raises."""
    if df is None:
        return None
    try:
        if hasattr(df, 'empty') and df.empty:
            return None
        return df.to_json(orient='records', date_format='iso')
    except Exception:
        return None


def _safe_json(obj: Any) -> Optional[str]:
    """JSON-encode anything, with a default=str fallback for datetimes etc."""
    if obj is None:
        return None
    try:
        return json.dumps(obj, default=str, sort_keys=True)
    except Exception:
        return None


def _quotes_to_json(quotes: dict) -> Optional[str]:
    """quotes dict is {symbol: quote_dict}. Strip to essential fields to keep
    JSON size manageable — full quote has many redundant fields."""
    if not quotes:
        return None
    try:
        slim = {}
        for sym, q in quotes.items():
            if not q:
                continue
            slim[sym] = {
                'last':   q.get('last'),
                'bid':    q.get('bid'),
                'ask':    q.get('ask'),
                'mark':   q.get('mark'),
                'mid':    q.get('mid'),
                'volume': q.get('volume'),
            }
        return json.dumps(slim, default=str)
    except Exception:
        return None


def _int_or_none(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(bool(v)) if isinstance(v, bool) else int(v)
    except (TypeError, ValueError):
        return None


def _float_or_none(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def archive_snapshot(
    snapshot:         dict,
    regime_state:     Any,
    term_structure:   Optional[dict] = None,
    macro_state:      Any = None,
    cycle_name:       Optional[str] = None,
    version_id:       Optional[int] = None,
    injected_symbols: Optional[dict] = None,
    universe_size:    Optional[int] = None,
) -> Optional[int]:
    """
    Persist a full cycle snapshot. Returns cycle_id for use in archive_decisions.

    Args:
        snapshot:         dict from data_collector.collect_snapshot()
        regime_state:     RegimeState dataclass from regime_engine.evaluate_regime()
        term_structure:   dict from data_collector.get_vix_term_structure()
        macro_state:      MacroState from macro_sentinel.evaluate_macro() (or None)
        cycle_name:       'premarket' | 'open' | 'mid_morning' | etc.
        version_id:       From bot_version.get_or_create_version_id()
        injected_symbols: {symbol: source} from top-of-funnel scanners
        universe_size:    Total symbols evaluated this cycle

    Best-effort: returns None on any failure, never crashes the bot.
    """
    try:
        _init_db()
        if not _init_done:
            return None

        now_utc       = datetime.now(timezone.utc)
        cycle_time    = now_utc.isoformat()
        # trading_date in ET (matches how the bot thinks about cycles)
        try:
            import pytz
            trading_date = now_utc.astimezone(pytz.timezone('America/New_York')).strftime('%Y-%m-%d')
        except Exception:
            trading_date = now_utc.strftime('%Y-%m-%d')

        ts = term_structure or {}

        # Regime state may be None or partially populated. Use getattr defensively
        # since RegimeState fields can vary between code versions.
        def rs(name, default=None):
            return getattr(regime_state, name, default) if regime_state else default

        # Macro state similarly
        def ms(name, default=None):
            return getattr(macro_state, name, default) if macro_state else default

        # Injected symbols payload
        injected_count = len(injected_symbols or {})
        injected_json  = _safe_json(injected_symbols) if injected_symbols else None

        # Fear & Greed
        fg = snapshot.get('fear_greed', {}) if snapshot else {}
        fg_score = _int_or_none(fg.get('score'))
        fg_label = fg.get('label') if isinstance(fg, dict) else None

        with _lock:
            conn = sqlite3.connect(DB_PATH, timeout=10)
            try:
                cur = conn.execute('''
                    INSERT INTO cycle_snapshots (
                        version_id, cycle_time, cycle_name, trading_date,
                        vixy, vixy_30d_avg, vixy_30d_std, vix_zscore,
                        vix9d, vix_30d, vix3m, term_signal, halt_entries,
                        regime, previous_regime, position_size, in_pause, mean_reversion_signal,
                        macro_score, macro_warning, macro_override,
                        fear_greed_score, fear_greed_label,
                        universe_size, injected_count, injected_json,
                        quotes_json, uw_darkpool_json
                    ) VALUES (?,?,?,?, ?,?,?,?, ?,?,?,?,?, ?,?,?,?,?, ?,?,?, ?,?, ?,?,?, ?,?)
                ''', (
                    version_id,
                    cycle_time,
                    cycle_name,
                    trading_date,

                    _float_or_none(rs('vixy')),
                    _float_or_none(rs('vixy_30d_avg')),
                    _float_or_none(rs('vixy_30d_std')),
                    _float_or_none(rs('vix_zscore')),

                    _float_or_none(ts.get('vix9d')),
                    _float_or_none(ts.get('vix')),
                    _float_or_none(ts.get('vix3m')),
                    ts.get('signal'),
                    _int_or_none(ts.get('halt_entries')),

                    rs('regime'),
                    rs('previous_regime'),
                    _float_or_none(rs('position_size')),
                    _int_or_none(rs('in_pause')),
                    _int_or_none(rs('mean_reversion_signal')),

                    _float_or_none(ms('score')),
                    ms('warning'),
                    _int_or_none(rs('macro_override')),

                    fg_score,
                    fg_label,

                    universe_size,
                    injected_count,
                    injected_json,

                    _quotes_to_json(snapshot.get('quotes', {})) if snapshot else None,
                    _df_to_json(snapshot.get('uw_darkpool')) if snapshot else None,
                ))
                cycle_id = cur.lastrowid

                # Per-symbol UW flow rows
                uw_flow_dict = (snapshot or {}).get('uw_flow', {})
                if isinstance(uw_flow_dict, dict):
                    flow_rows = []
                    for sym, df in uw_flow_dict.items():
                        flow_json = _df_to_json(df)
                        if not flow_json:
                            continue
                        try:
                            row_count = len(df) if df is not None else 0
                        except Exception:
                            row_count = 0
                        flow_rows.append((cycle_id, version_id, sym, flow_json, row_count))

                    if flow_rows:
                        conn.executemany('''
                            INSERT INTO cycle_uw_flow
                            (cycle_id, version_id, symbol, flow_json, row_count)
                            VALUES (?, ?, ?, ?, ?)
                        ''', flow_rows)

                conn.commit()
                print(f"  [snapshot_capture] Archived cycle #{cycle_id} "
                      f"({cycle_name}, v{version_id})")
                return cycle_id
            finally:
                conn.close()

    except Exception as e:
        print(f"  [snapshot_capture] WARNING: archive_snapshot failed: {e}")
        return None


def archive_decisions(
    cycle_id:        int,
    version_id:      Optional[int] = None,
    scored_results:  Optional[list] = None,
    passed_risk:     Optional[list] = None,
    blocked:         Optional[dict] = None,
    candidates:      Optional[list] = None,
    traded:          Optional[list] = None,
    injected_symbols: Optional[dict] = None,
) -> int:
    """
    Persist per-ticker decisions for a cycle. Best-effort, never raises.
    Returns the number of decision rows written.

    Args:
        cycle_id:         From archive_snapshot()
        version_id:       From bot_version.get_or_create_version_id()
        scored_results:   List of StockScore from run_scoring_cycle().all_scores
        passed_risk:      List of symbols that passed risk_manager
        blocked:          {symbol: ['blocker1', 'blocker2', ...]}
        candidates:       List of StockScore that became trade candidates
        traded:           List of dicts from execute_trades(): symbol, shares, price, success, ...
        injected_symbols: {symbol: source} from top-of-funnel
    """
    if not cycle_id:
        return 0
    try:
        _init_db()
        if not _init_done:
            return 0

        passed_set = set(passed_risk or [])
        blocked = blocked or {}
        injected_symbols = injected_symbols or {}
        scored_results = scored_results or []
        candidates = candidates or []
        traded = traded or []

        candidate_symbols = {c.symbol for c in candidates if hasattr(c, 'symbol')}

        # Build symbol → trade-result lookup
        trade_by_symbol = {}
        for tr in traded:
            if isinstance(tr, dict) and tr.get('symbol'):
                trade_by_symbol[tr['symbol']] = tr

        # Get cycle metadata for denormalized fields (cycle_time, trading_date)
        conn = sqlite3.connect(DB_PATH, timeout=10)
        try:
            row = conn.execute(
                'SELECT cycle_time, trading_date FROM cycle_snapshots WHERE cycle_id = ?',
                (cycle_id,)
            ).fetchone()
            if not row:
                print(f"  [snapshot_capture] WARNING: cycle_id {cycle_id} not found")
                return 0
            cycle_time, trading_date = row

            # Build complete symbol set: scored + blocked + traded
            all_symbols = set()
            for s in scored_results:
                if hasattr(s, 'symbol'):
                    all_symbols.add(s.symbol)
            all_symbols.update(blocked.keys())
            all_symbols.update(trade_by_symbol.keys())
            all_symbols.update(injected_symbols.keys())

            # Lookup table for scored results
            scored_by_symbol = {
                s.symbol: s for s in scored_results if hasattr(s, 'symbol')
            }

            rows = []
            for sym in sorted(all_symbols):
                s = scored_by_symbol.get(sym)
                blockers = blocked.get(sym, [])
                trade = trade_by_symbol.get(sym)
                source = injected_symbols.get(sym, 'universe')

                rows.append((
                    cycle_id,
                    version_id,
                    cycle_time,
                    trading_date,
                    sym,
                    source,

                    1 if sym in passed_set else 0,
                    ', '.join(blockers) if blockers else None,

                    _float_or_none(getattr(s, 'total_score', None)) if s else None,
                    _int_or_none(getattr(s, 'qualifies', None)) if s else None,
                    getattr(s, 'direction', None) if s else None,
                    _safe_json(getattr(s, 'signals', None)) if s else None,
                    _safe_json(getattr(s, 'weighted', None)) if s else None,

                    1 if sym in candidate_symbols else 0,
                    1 if trade else 0,
                    _int_or_none(trade.get('success')) if trade else None,
                    _int_or_none(trade.get('shares')) if trade else None,
                    _float_or_none(trade.get('price')) if trade else None,
                    trade.get('reason_skip') if trade else None,
                ))

            if rows:
                conn.executemany('''
                    INSERT INTO cycle_decisions (
                        cycle_id, version_id, cycle_time, trading_date,
                        symbol, source,
                        passed_risk, blockers,
                        total_score, qualifies, direction, signals_json, weighted_json,
                        was_candidate, attempted_trade, trade_succeeded,
                        trade_shares, trade_price, trade_reason_skip
                    ) VALUES (?,?,?,?, ?,?, ?,?, ?,?,?,?,?, ?,?,?, ?,?,?)
                ''', rows)
                conn.commit()
                print(f"  [snapshot_capture] Logged {len(rows)} decisions for cycle #{cycle_id}")

            return len(rows)
        finally:
            conn.close()

    except Exception as e:
        print(f"  [snapshot_capture] WARNING: archive_decisions failed: {e}")
        return 0


# ── Read-side helpers ─────────────────────────────────────────────────────────

def db_path() -> str:
    return DB_PATH


def summary_stats():
    """Print a quick summary of captured data. Useful for verification."""
    if not os.path.exists(DB_PATH):
        print(f"No DB at {DB_PATH} yet — nothing captured.")
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        try:
            versions = conn.execute("SELECT COUNT(*) FROM bot_versions").fetchone()[0]
        except sqlite3.OperationalError:
            versions = 0
        cycles = conn.execute("SELECT COUNT(*) FROM cycle_snapshots").fetchone()[0]
        decisions = conn.execute("SELECT COUNT(*) FROM cycle_decisions").fetchone()[0]
        flow_rows = conn.execute("SELECT COUNT(*) FROM cycle_uw_flow").fetchone()[0]
        first = conn.execute("SELECT MIN(cycle_time) FROM cycle_snapshots").fetchone()[0]
        last  = conn.execute("SELECT MAX(cycle_time) FROM cycle_snapshots").fetchone()[0]
        traded = conn.execute(
            "SELECT COUNT(*) FROM cycle_decisions WHERE trade_succeeded = 1"
        ).fetchone()[0]
        candidates = conn.execute(
            "SELECT COUNT(*) FROM cycle_decisions WHERE was_candidate = 1"
        ).fetchone()[0]

        print(f"snapshot_capture summary:")
        print(f"  DB:               {DB_PATH}")
        print(f"  Bot versions:     {versions}")
        print(f"  Cycle snapshots:  {cycles}")
        print(f"  UW flow rows:     {flow_rows}")
        print(f"  Decisions:        {decisions}")
        print(f"  Candidates:       {candidates}")
        print(f"  Trades (success): {traded}")
        print(f"  First cycle:      {first}")
        print(f"  Last cycle:       {last}")
    finally:
        conn.close()


if __name__ == '__main__':
    summary_stats()