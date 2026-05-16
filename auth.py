import schwab
import os
from dotenv import load_dotenv

# Load your API keys from .env file
load_dotenv()

APP_KEY = os.getenv('SCHWAB_APP_KEY')
APP_SECRET = os.getenv('SCHWAB_APP_SECRET')
CALLBACK_URL = os.getenv('SCHWAB_CALLBACK_URL')

# Switch between 'token.json' (live) and 'paper_token.json' (paper)
PAPER_TRADING = True

TOKEN_PATH = 'paper_token.json' if PAPER_TRADING else 'token.json'

def authenticate():
    client = schwab.auth.easy_client(
        api_key=APP_KEY,
        app_secret=APP_SECRET,
        callback_url=CALLBACK_URL,
        token_path=TOKEN_PATH
    )
    return client, PAPER_TRADING

if __name__ == '__main__':
    mode = "📄 PAPER TRADING" if PAPER_TRADING else "💰 LIVE TRADING"
    print(f"Connecting to Schwab in {mode} mode...")
    client, paper = authenticate()
    
    # Test the connection by fetching your account numbers
    response = client.get_account_numbers()
    accounts = response.json()
    
    print("✅ Connected successfully!")
    print(f"Found {len(accounts)} account(s):")
    for account in accounts:
        print(f"  - Account: {account['accountNumber']}")
    print(f"\n⚠️  Mode: {mode}")
    print("To switch modes, change PAPER_TRADING = True/False in auth.py")
