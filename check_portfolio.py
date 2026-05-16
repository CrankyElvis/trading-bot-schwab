import schwab
import json
from auth import authenticate

client = authenticate()

# Get account numbers
accounts = client.get_account_numbers().json()
account_hash = accounts[0]['hashValue']

# Get full account details including positions
response = client.get_account(
    account_hash,
    fields=[schwab.client.Client.Account.Fields.POSITIONS]
)

data = response.json()

# Display balance
balance = data['securitiesAccount']['currentBalances']
print("\n💰 ACCOUNT BALANCES:")
print(f"  Cash Available:     ${balance.get('cashAvailableForTrading', 0):,.2f}")
print(f"  Account Value:      ${balance.get('liquidationValue', 0):,.2f}")
print(f"  Buying Power:       ${balance.get('buyingPower', 0):,.2f}")

# Display positions
positions = data['securitiesAccount'].get('positions', [])
print(f"\n📊 CURRENT POSITIONS ({len(positions)} total):")
if positions:
    for pos in positions:
        inst = pos['instrument']
        symbol = inst.get('symbol', 'N/A')
        qty = pos.get('longQuantity', 0)
        avg_price = pos.get('averagePrice', 0)
        market_val = pos.get('marketValue', 0)
        gain_loss = pos.get('currentDayProfitLoss', 0)
        print(f"  {symbol:<10} Qty: {qty:<8} Avg: ${avg_price:<10.2f} Value: ${market_val:<12.2f} Day P/L: ${gain_loss:.2f}")
else:
    print("  No open positions found")