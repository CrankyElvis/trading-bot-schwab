import sys
sys.path.insert(0, '.')
from data_collector import (_sec_lookup_cik, _sec_fetch_submissions,
                              _sec_fetch_and_parse_form4, SEC_HEADERS)
import requests
from datetime import datetime, timedelta

sym = 'AAPL'

print('=== Step 1: CIK lookup ===')
cik = _sec_lookup_cik(sym)
print(f'  CIK for {sym}: {cik}')

print('=== Step 2: Fetch submissions ===')
subs = _sec_fetch_submissions(sym, cik)
print(f'  Got submissions: {subs is not None}')
form4s = []
if subs:
    recent = subs.get('filings', {}).get('recent', {})
    forms = recent.get('form', [])
    dates = recent.get('filingDate', [])
    accessions = recent.get('accessionNumber', [])
    print(f'  Total filings: {len(forms)}')
    cutoff = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    print(f'  Cutoff date: {cutoff}')
    form4s = [(f,d,a) for f,d,a in zip(forms,dates,accessions) if f == '4']
    print(f'  All Form 4s: {len(form4s)}')
    for f,d,a in form4s[:5]:
        print(f'    {d}  acc={a}  in_window={d >= cutoff}')

print('=== Step 3: Try fetch + parse first Form 4 ===')
if form4s:
    f,d,a = form4s[0]
    result = _sec_fetch_and_parse_form4(cik, a, sym, d)
    print(f'  Result: {result}')

print('=== Step 4: Raw URL test ===')
if form4s:
    f,d,a = form4s[0]
    cik_unpadded = str(int(cik))
    accession_no_dashes_only = a.replace('-', '')
    url = f'https://www.sec.gov/Archives/edgar/data/{cik_unpadded}/{accession_no_dashes_only}/primary_doc.xml'
    print(f'  URL: {url}')
    r = requests.get(url, headers=SEC_HEADERS, timeout=15)
    ct = r.headers.get('Content-Type')
    print(f'  HTTP status: {r.status_code}')
    print(f'  Content-Type: {ct}')
    print(f'  Length: {len(r.text)} chars')
    print(f'  First 400 chars:')
    print(r.text[:400])
