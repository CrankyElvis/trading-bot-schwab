"""
bot_version.py
─────────────────────────────────────────────────────────────────────────────
Computes a deterministic version fingerprint by SHA256-hashing every .py file
in the bot's root directory at startup. Persists the manifest to snapshots.db
so every snapshot, decision, and fill can foreign-key back to "which version
of the code produced this row?"

This is the primary mechanism for distinguishing market effects from code
effects when analyzing live trading behavior. If insider-signal contribution
suddenly improves on 2026-05-27, was it because we fixed the SEC EDGAR
User-Agent, or did insider buying coincidentally pick up? The version_id on
each row makes this answerable.

Design choices:
  - Hashes ALL .py files in BOT_DIR (non-recursive), not selectively. That
    way new files added later are captured automatically without code change.
  - Excludes the venv, __pycache__, and test_* / *_test files. Tests don't
    affect production behavior.
  - One row per (file_hash_set) — same hashes = same version_id. Restart with
    no code change returns the existing version_id.
  - Single source of truth: snapshots.db, same DB used by snapshot_capture.
    Avoids two-DB-coordination problems.

Usage:
    from bot_version import get_or_create_version_id
    version_id = get_or_create_version_id()
    # Pass version_id when writing snapshots, decisions, fills
"""

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Configuration ─────────────────────────────────────────────────────────────

BOT_DIR  = os.environ.get('BOT_DIR', '/root/trading-bot')
DB_PATH  = os.environ.get('SNAPSHOTS_DB', '/root/trading-bot/data/snapshots.db')

# Files we don't want in the version fingerprint
EXCLUDE_PREFIXES = ('test_', '_test')
EXCLUDE_SUFFIXES = ('_test.py',)
EXCLUDE_DIRS     = {'venv', '__pycache__', '.git', 'data', 'validation'}

_lock = threading.Lock()
_init_done = False
_cached_version_id: Optional[int] = None


# ── Schema ────────────────────────────────────────────────────────────────────

def _init_db():
    """Create bot_versions table if missing. Idempotent."""
    global _init_done
    if _init_done:
        return
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        # Manifest hash is SHA256 of the sorted JSON of {filename: file_hash},
        # giving a single string that uniquely identifies a code configuration.
        conn.execute('''
            CREATE TABLE IF NOT EXISTS bot_versions (
                version_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                manifest_hash   TEXT NOT NULL UNIQUE,
                file_count      INTEGER NOT NULL,
                files_json      TEXT NOT NULL,
                first_seen_at   TEXT NOT NULL,
                last_seen_at    TEXT NOT NULL
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_versions_manifest '
                     'ON bot_versions(manifest_hash)')
        conn.commit()
        conn.close()
        _init_done = True
    except Exception as e:
        print(f"  [bot_version] WARNING: DB init failed: {e}")


# ── File hashing ──────────────────────────────────────────────────────────────

def _should_include(filename: str) -> bool:
    """Filters out test files and non-.py files."""
    if not filename.endswith('.py'):
        return False
    if any(filename.startswith(p) for p in EXCLUDE_PREFIXES):
        return False
    if any(filename.endswith(s) for s in EXCLUDE_SUFFIXES):
        return False
    return True


def _hash_file(path: Path) -> str:
    """SHA256 of file contents, first 16 hex chars (64 bits, plenty of entropy)."""
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception as e:
        return f'ERR:{e}'


def compute_manifest() -> dict:
    """
    Walks BOT_DIR (non-recursive) and hashes every production .py file.
    Returns {filename: hash} dict, sorted by filename.
    """
    bot_path = Path(BOT_DIR)
    if not bot_path.is_dir():
        print(f"  [bot_version] WARNING: BOT_DIR {BOT_DIR} not a directory")
        return {}

    manifest = {}
    # Non-recursive — only top-level .py files. Excludes ./validation/, ./data/, etc.
    for entry in sorted(bot_path.iterdir()):
        if entry.is_dir():
            continue
        if not _should_include(entry.name):
            continue
        manifest[entry.name] = _hash_file(entry)

    return manifest


def manifest_hash(manifest: dict) -> str:
    """Returns SHA256 of the canonical JSON of the manifest. Single string
    uniquely identifying this code configuration."""
    canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:24]


# ── Public API ────────────────────────────────────────────────────────────────

def get_or_create_version_id(force_recompute: bool = False) -> Optional[int]:
    """
    Returns the version_id for the current code configuration.

    Behavior:
      - If this hash exists in bot_versions: returns its existing version_id
        and updates last_seen_at to now.
      - If this hash is new: inserts a row and returns the new version_id.
      - On any failure: returns None (caller should still proceed without
        version tracking rather than crash).

    Cached in-process — bot startup computes once, all later calls return
    the cached value unless force_recompute=True.
    """
    global _cached_version_id
    if _cached_version_id is not None and not force_recompute:
        return _cached_version_id

    try:
        _init_db()
        if not _init_done:
            return None

        manifest = compute_manifest()
        if not manifest:
            print("  [bot_version] WARNING: empty manifest, no version tracking this run")
            return None

        m_hash = manifest_hash(manifest)
        now_iso = datetime.now(timezone.utc).isoformat()

        with _lock:
            conn = sqlite3.connect(DB_PATH, timeout=10)
            try:
                row = conn.execute(
                    'SELECT version_id FROM bot_versions WHERE manifest_hash = ?',
                    (m_hash,)
                ).fetchone()

                if row:
                    version_id = row[0]
                    conn.execute(
                        'UPDATE bot_versions SET last_seen_at = ? WHERE version_id = ?',
                        (now_iso, version_id)
                    )
                else:
                    cur = conn.execute('''
                        INSERT INTO bot_versions
                        (manifest_hash, file_count, files_json, first_seen_at, last_seen_at)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        m_hash,
                        len(manifest),
                        json.dumps(manifest, sort_keys=True),
                        now_iso,
                        now_iso,
                    ))
                    version_id = cur.lastrowid
                    print(f"  [bot_version] NEW version_id={version_id} "
                          f"({len(manifest)} files, hash={m_hash[:12]}...)")

                conn.commit()
            finally:
                conn.close()

        _cached_version_id = version_id
        return version_id

    except Exception as e:
        print(f"  [bot_version] WARNING: get_or_create_version_id failed: {e}")
        return None


def get_current_version_info() -> dict:
    """
    Returns a summary of the current version for logging/debugging.
    """
    try:
        manifest = compute_manifest()
        m_hash = manifest_hash(manifest) if manifest else None
        return {
            'version_id':       _cached_version_id,
            'manifest_hash':    m_hash,
            'file_count':       len(manifest),
            'bot_dir':          BOT_DIR,
        }
    except Exception as e:
        return {'error': str(e)}


def diff_against(other_version_id: int) -> dict:
    """
    Returns the file-level diff between the current code and a past version_id.
    Useful for "what changed between version 4 and version 7?"
    """
    try:
        _init_db()
        if not _init_done:
            return {'error': 'DB unavailable'}

        current_manifest = compute_manifest()
        conn = sqlite3.connect(DB_PATH, timeout=10)
        try:
            row = conn.execute(
                'SELECT files_json FROM bot_versions WHERE version_id = ?',
                (other_version_id,)
            ).fetchone()
            if not row:
                return {'error': f'version_id {other_version_id} not found'}
            other_manifest = json.loads(row[0])
        finally:
            conn.close()

        all_files = set(current_manifest.keys()) | set(other_manifest.keys())
        changed = []
        added = []
        removed = []
        for f in sorted(all_files):
            in_current = f in current_manifest
            in_other = f in other_manifest
            if in_current and in_other:
                if current_manifest[f] != other_manifest[f]:
                    changed.append({
                        'file':    f,
                        'old':     other_manifest[f],
                        'new':     current_manifest[f],
                    })
            elif in_current:
                added.append(f)
            else:
                removed.append(f)

        return {
            'against_version_id': other_version_id,
            'changed':            changed,
            'added':              added,
            'removed':            removed,
        }
    except Exception as e:
        return {'error': str(e)}


# ── Read-side helpers ─────────────────────────────────────────────────────────

def list_versions(limit: int = 20) -> list:
    """Returns a list of recent versions for inspection."""
    try:
        _init_db()
        if not _init_done:
            return []
        conn = sqlite3.connect(DB_PATH, timeout=10)
        try:
            rows = conn.execute('''
                SELECT version_id, manifest_hash, file_count,
                       first_seen_at, last_seen_at
                FROM bot_versions
                ORDER BY version_id DESC
                LIMIT ?
            ''', (limit,)).fetchall()
            return [
                {
                    'version_id':     r[0],
                    'manifest_hash':  r[1],
                    'file_count':     r[2],
                    'first_seen_at':  r[3],
                    'last_seen_at':   r[4],
                }
                for r in rows
            ]
        finally:
            conn.close()
    except Exception as e:
        print(f"  [bot_version] list_versions failed: {e}")
        return []


def summary_stats():
    """Print a quick summary. Useful for sanity checks from the CLI."""
    print(f"bot_version summary:")
    print(f"  BOT_DIR:  {BOT_DIR}")
    print(f"  DB:       {DB_PATH}")

    manifest = compute_manifest()
    m_hash = manifest_hash(manifest) if manifest else 'EMPTY'
    print(f"  Current manifest:")
    print(f"    file_count:    {len(manifest)}")
    print(f"    manifest_hash: {m_hash}")
    if manifest:
        print(f"    Files:")
        for f, h in manifest.items():
            print(f"      {h}  {f}")

    versions = list_versions(limit=10)
    print(f"\n  Recent versions in DB ({len(versions)}):")
    for v in versions:
        print(f"    #{v['version_id']:>3}  {v['manifest_hash'][:12]}...  "
              f"{v['file_count']} files  first={v['first_seen_at'][:19]}  "
              f"last={v['last_seen_at'][:19]}")


if __name__ == '__main__':
    summary_stats()
    print()
    vid = get_or_create_version_id()
    print(f"\nCurrent version_id: {vid}")