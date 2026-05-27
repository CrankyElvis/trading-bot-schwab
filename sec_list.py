import requests
from data_collector import SEC_HEADERS

# Get the filing index — lists all files for this accession
url = "https://www.sec.gov/Archives/edgar/data/320193/000114036126020871/"
r = requests.get(url, headers=SEC_HEADERS, timeout=15)
print(f"HTTP {r.status_code}")
print(r.text[:2500])
