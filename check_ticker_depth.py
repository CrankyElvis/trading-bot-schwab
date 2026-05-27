"""
check_ticker_depth.py
─────────────────────
Checks the options_history.db to see which tickers have how much data.
Used to decide which tickers are worth downloading more data for.

Usage: python check_ticker_depth.py
"""

import sqlite3
import os

DB_PATH = r'C:\trading-bot\data\options_history.db'

def main():
    if not os.path.exists(DB_PATH):
        print(f"DB not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)

    # All tickers, all sides
    print("=" * 78)
    print("All tickers in DB (puts and calls separately):")
    print("=" * 78)
    print(f"{'Symbol':<8}{'Side':<8}{'Records':>10}{'First Date':>14}{'Last Date':>14}{'Years':>8}")
    print("-" * 78)

    rows = conn.execute("""
        SELECT symbol, side, COUNT(*), MIN(trade_date), MAX(trade_date)
        FROM options_history
        GROUP BY symbol, side
        ORDER BY symbol, side
    """).fetchall()

    from datetime import datetime
    for symbol, side, count, first, last in rows:
        try:
            d1 = datetime.strptime(first, '%Y-%m-%d')
            d2 = datetime.strptime(last, '%Y-%m-%d')
            years = (d2 - d1).days / 365.25
        except Exception:
            years = 0
        print(f"{symbol:<8}{side:<8}{count:>10}{first:>14}{last:>14}{years:>7.1f}y")

    # Focus on candidate tickers
    print()
    print("=" * 78)
    print("Candidate tickers for 5-year wheel backtest:")
    print("=" * 78)
    candidates = ['AAPL', 'MSFT', 'JPM', 'TSLA', 'NVDA', 'GOOGL', 'META', 'SPY']

    for sym in candidates:
        result = conn.execute("""
            SELECT side, COUNT(*), MIN(trade_date), MAX(trade_date)
            FROM options_history
            WHERE symbol = ?
            GROUP BY side
        """, (sym,)).fetchall()

        if not result:
            print(f"  {sym:<6}: NOT IN DB")
            continue

        by_side = {row[0]: row for row in result}
        put = by_side.get('put')
        call = by_side.get('call')

        put_str = f"puts={put[1]} ({put[2]}->{put[3]})" if put else "puts=NONE"
        call_str = f"calls={call[1]} ({call[2]}->{call[3]})" if call else "calls=NONE"

        # Verdict
        has_5yr_puts = put and put[2] < '2022-01-01'
        has_5yr_calls = call and call[2] < '2022-01-01'

        if has_5yr_puts and has_5yr_calls:
            verdict = "READY (5yr puts + calls)"
        elif has_5yr_puts and not call:
            verdict = "NEED CALLS (5yr puts done)"
        elif put and not call:
            verdict = f"NEED CALLS + DEEPER PUTS"
        elif not put:
            verdict = "MISSING ENTIRELY"
        else:
            verdict = "PARTIAL"

        print(f"  {sym:<6}: {put_str:<35} {call_str:<35} -> {verdict}")

    conn.close()


if __name__ == '__main__':
    main()