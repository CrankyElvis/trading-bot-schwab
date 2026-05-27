from data_collector import score_sec_insider, get_sec_form4
import time

t = time.time()
score = score_sec_insider("AAPL", days=30)
f = get_sec_form4("AAPL", days=30)
elapsed = time.time() - t

print(f"AAPL: score={score}  buys={f['buy_count']}  sells={f['sell_count']}  filings={f['filing_count']}  ({elapsed:.1f}s)")
