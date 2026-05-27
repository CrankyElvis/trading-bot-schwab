"""
run_premarket_scan.py
Manually trigger the pre-market scanner.

Runs the same scan that the 6am scheduled cycle would have run.
Writes the watchlist file. Bot will pick it up on the next cycle.

Usage on server:
    cd /root/trading-bot
    venv/bin/python3 run_premarket_scan.py
"""

from auth import authenticate
from pre_market_scanner import run_scan, load_watchlist, get_watchlist_candidates

print("Authenticating with Schwab...")
client, paper = authenticate()
print(f"Connected ({'PAPER' if paper else 'LIVE'})")

print("\nStarting pre-market scan (this takes 3-5 min)...")
run_scan(client)
print("\nScan complete.")

# Display results
wl = load_watchlist()
if not wl:
    print("ERROR: No watchlist file written")
else:
    print(f"\n{'='*60}")
    print(f"  WATCHLIST")
    print(f"{'='*60}")
    print(f"  Scanned at:      {wl.get('scanned_at', 'unknown')}")
    print(f"  Symbols scanned: {wl.get('symbols_scanned', 0)}")
    print(f"  Qualified:       {wl.get('qualified', 0)}")
    print()

    candidates = get_watchlist_candidates(min_score=0.70, max_n=25)
    if not candidates:
        print("  No candidates above 0.70 threshold.")
    else:
        print(f"  {'Symbol':<8} {'Score':>6}  {'Dir':<10} {'RSI':>5}")
        print(f"  {'-'*40}")
        for c in candidates:
            print(f"  {c['symbol']:<8} {c['score']:>6.3f}  "
                  f"{c.get('direction','neutral'):<10} {c.get('rsi',0):>5.1f}")
    print(f"{'='*60}")