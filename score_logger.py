"""
score_logger.py
─────────────────────────────────────────────────────────────────────────────
Persists every scoring cycle to scores_live.db so we can analyze how scores
relate to actual outcomes over time.

This is OUT-OF-SAMPLE evidence collection -- the only honest way to validate
whether the signal stack has real predictive edge.

Design:
  - Separate DB from options_history.db (no collision risk)
  - WAL mode for concurrent reads from analysis scripts while bot writes
  - Schema-versioned for future evolution
  - Fails silently with a warning -- never crashes the live bot

Usage from flow_momentum.py:
    from score_logger import log_cycle
    log_cycle(result, regime, vix)

Schema:
    scores_live.db
      scoring_cycles      one row per (cycle_id) -- when the cycle ran
      score_observations  one row per (cycle_id, ticker) -- every scored ticker
"""

import os
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

# Default DB location -- override with env var SCORE_LOG_DB if needed
DEFAULT_DB_PATH = '/root/trading-bot/data/scores_live.db'
DB_PATH = os.environ.get('SCORE_LOG_DB', DEFAULT_DB_PATH)

# Thread safety -- SQLite handles concurrent writers via WAL, but we also
# serialize within-process to avoid edge-case races
_lock = threading.Lock()
_init_done = False


# ── Schema initialization ─────────────────────────────────────────────────────

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

        # Cycle metadata: one row per scoring cycle (every ~4 hours when bot runs)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS scoring_cycles (
                cycle_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_time      TEXT NOT NULL,
                regime          TEXT,
                vix             REAL,
                portfolio_value REAL,
                min_score       REAL,
                max_candidates  INTEGER,
                n_scored        INTEGER,
                n_qualified     INTEGER,
                n_candidates    INTEGER,
                fg_score        INTEGER,
                fg_modifier     REAL,
                pc_ratio        REAL,
                pc_modifier     REAL,
                created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Per-ticker observations: one row per (cycle_id, ticker)
        # Captures the FULL signal breakdown, the modified total, and what
        # the bot decided to do with this ticker.
        conn.execute('''
            CREATE TABLE IF NOT EXISTS score_observations (
                obs_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id        INTEGER NOT NULL,
                cycle_time      TEXT NOT NULL,
                ticker          TEXT NOT NULL,
                total_score     REAL NOT NULL,
                qualifies       INTEGER NOT NULL,
                is_candidate    INTEGER NOT NULL,
                direction       TEXT,
                regime          TEXT,
                vix             REAL,

                -- Individual signal raw scores
                sig_sweep_flow  REAL,
                sig_dark_pool   REAL,
                sig_politician  REAL,
                sig_insider     REAL,
                sig_price_rvol  REAL,
                sig_gex         REAL,
                sig_market_tide REAL,
                sig_sector_tide REAL,
                sig_etf_flow    REAL,

                -- Full signal JSON (future-proof: new signals get stored without schema change)
                signals_json    TEXT,
                weighted_json   TEXT,

                FOREIGN KEY (cycle_id) REFERENCES scoring_cycles(cycle_id)
            )
        ''')

        # Index for fast (ticker, time) lookups when analyzing outcomes later
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_obs_ticker_time
                ON score_observations(ticker, cycle_time)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_obs_cycle
                ON score_observations(cycle_id)
        ''')

        conn.commit()
        conn.close()
        _init_done = True
    except Exception as e:
        # Logging is best-effort; never break the bot
        print(f"  [score_logger] WARNING: DB init failed: {e}")


# ── Public API ────────────────────────────────────────────────────────────────

def log_cycle(result, regime=None, vix=None, portfolio_value=None,
              fg_score=None, fg_modifier=None, pc_ratio=None, pc_modifier=None):
    """
    Persist a complete scoring cycle to scores_live.db.

    Args:
        result:          CycleResult from run_scoring_cycle() in flow_momentum.py
        regime:          Current regime ('flow', 'neutral', etc.)
        vix:             Current VIX level
        portfolio_value: Current portfolio value (drives max_candidates scaling)
        fg_score:        Fear & Greed numeric score (0-100)
        fg_modifier:     Fear & Greed score multiplier applied this cycle
        pc_ratio:        Put/Call ratio
        pc_modifier:     P/C score multiplier applied this cycle

    Never raises. Best-effort logging.
    """
    try:
        _init_db()
        if not _init_done:
            return   # init failed, skip logging this cycle

        with _lock:
            conn = sqlite3.connect(DB_PATH, timeout=10)
            try:
                # Import here to avoid circular dependency at module load time
                try:
                    from flow_momentum import MIN_SCORE, get_max_candidates
                    min_score = MIN_SCORE
                    max_candidates = get_max_candidates(portfolio_value or 0)
                except Exception:
                    min_score = None
                    max_candidates = None

                candidate_symbols = {c.symbol for c in result.candidates}

                # Insert cycle metadata
                cur = conn.execute('''
                    INSERT INTO scoring_cycles
                    (cycle_time, regime, vix, portfolio_value, min_score,
                     max_candidates, n_scored, n_qualified, n_candidates,
                     fg_score, fg_modifier, pc_ratio, pc_modifier)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ''', (
                    result.cycle_time,
                    regime or result.regime,
                    vix,
                    portfolio_value,
                    min_score,
                    max_candidates,
                    len(result.all_scores),
                    sum(1 for s in result.all_scores if s.qualifies),
                    len(result.candidates),
                    fg_score, fg_modifier, pc_ratio, pc_modifier,
                ))
                cycle_id = cur.lastrowid

                # Insert one row per scored ticker
                rows = []
                for s in result.all_scores:
                    sigs = s.signals or {}
                    rows.append((
                        cycle_id,
                        s.timestamp or result.cycle_time,
                        s.symbol,
                        float(s.total_score),
                        1 if s.qualifies else 0,
                        1 if s.symbol in candidate_symbols else 0,
                        s.direction,
                        regime or result.regime,
                        vix,
                        sigs.get('sweep_flow'),
                        sigs.get('dark_pool'),
                        sigs.get('politician'),
                        sigs.get('insider'),
                        sigs.get('price_rvol'),
                        sigs.get('gex'),
                        sigs.get('market_tide'),
                        sigs.get('sector_tide'),
                        sigs.get('etf_flow'),
                        json.dumps(s.signals, default=str) if s.signals else None,
                        json.dumps(s.weighted, default=str) if s.weighted else None,
                    ))

                conn.executemany('''
                    INSERT INTO score_observations
                    (cycle_id, cycle_time, ticker, total_score, qualifies,
                     is_candidate, direction, regime, vix,
                     sig_sweep_flow, sig_dark_pool, sig_politician, sig_insider,
                     sig_price_rvol, sig_gex, sig_market_tide, sig_sector_tide,
                     sig_etf_flow, signals_json, weighted_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ''', rows)

                conn.commit()
                print(f"  [score_logger] Logged cycle #{cycle_id}: "
                      f"{len(rows)} observations, "
                      f"{len(result.candidates)} candidates")
            finally:
                conn.close()
    except Exception as e:
        print(f"  [score_logger] WARNING: log_cycle failed: {e}")


# ── Read-side helpers (for analysis scripts) ──────────────────────────────────

def db_path() -> str:
    """Returns the active DB path so analysis scripts can connect."""
    return DB_PATH


def summary_stats():
    """Print a quick summary of logged data. Useful for sanity checks."""
    if not os.path.exists(DB_PATH):
        print(f"No DB at {DB_PATH} yet -- nothing logged.")
        return
    conn = sqlite3.connect(DB_PATH)
    cycles = conn.execute("SELECT COUNT(*) FROM scoring_cycles").fetchone()[0]
    obs    = conn.execute("SELECT COUNT(*) FROM score_observations").fetchone()[0]
    first  = conn.execute("SELECT MIN(cycle_time) FROM scoring_cycles").fetchone()[0]
    last   = conn.execute("SELECT MAX(cycle_time) FROM scoring_cycles").fetchone()[0]
    qual   = conn.execute("SELECT COUNT(*) FROM score_observations WHERE qualifies=1").fetchone()[0]
    cand   = conn.execute("SELECT COUNT(*) FROM score_observations WHERE is_candidate=1").fetchone()[0]
    conn.close()
    print(f"score_logger summary:")
    print(f"  DB:           {DB_PATH}")
    print(f"  Cycles:       {cycles}")
    print(f"  Observations: {obs}")
    print(f"  Qualified:    {qual}  ({100*qual/max(obs,1):.1f}%)")
    print(f"  Candidates:   {cand}  ({100*cand/max(obs,1):.1f}%)")
    print(f"  First cycle:  {first}")
    print(f"  Last cycle:   {last}")


if __name__ == '__main__':
    summary_stats()
