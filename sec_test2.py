from data_collector import score_sec_insider, get_sec_form4
import time

# Test 5 tickers, varied profiles
for sym in ["AAPL", "NVDA", "TSLA", "META", "MSFT", "PLTR", "GME"]:
    t = time.time()
    score = score_sec_insider(sym, days=30)
    f = get_sec_form4(sym, days=30)
    elapsed = time.time() - t
    print(f"{sym:6s}: score={score:.2f}  buys={f['buy_count']:2d}  sells={f['sell_count']:2d}  filings={f['filing_count']:2d}  ({elapsed:.1f}s)")
