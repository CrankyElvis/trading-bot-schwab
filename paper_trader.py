import json
import os
from datetime import datetime
from auth import authenticate

PAPER_PORTFOLIO_FILE = 'paper_portfolio.json'

# ── Fee Constants ─────────────────────────────────────────────────────────────
OPTIONS_COMMISSION_PER_CONTRACT = 0.65   # Schwab options rate
SPREAD_COST_PCT = 0.0005                 # 0.05% default (overridden by liquidity tier)

# ── Liquidity tiers — bid/ask spread model ────────────────────────────────────
# Tier 1 (mega-cap ETFs): SPY QQQ IWM GLD TLT — spread ~0.01%
# Tier 2 (large-cap):     AAPL MSFT NVDA AMZN META TSLA etc — spread ~0.03%
# Tier 3 (mid-cap):       most individual stocks — spread ~0.08%
# Tier 4 (small/volatile): MSTR MEME stocks LCID RIVN etc — spread ~0.15%
_TIER1 = {'SPY','QQQ','IWM','DIA','GLD','GDX','TLT','SLV','IAU','EEM',
           'EFA','XLF','XLK','XLE','XLV','XLI','XLP','XLU','XLB','XLC',
           'XLRE','XLY','HYG','LQD','VTI','VNQ','SMH','SOXX','IBB','MBB',
           'VIXY','VXX','UVXY','SQQQ','SPXU','USO','UNG','VTIP','SCHP'}
_TIER2 = {'AAPL','MSFT','NVDA','AMZN','META','TSLA','GOOGL','GOOG','AVGO',
           'JPM','BAC','GS','MS','WFC','C','BLK','SCHW','COST','HD','LOW',
           'UNH','LLY','JNJ','PFE','ABBV','MRK','TMO','DHR','ISRG','VRTX',
           'V','MA','PYPL','NDAQ','CME','ICE','SPGI','MCO','BK','STT',
           'AMGN','GILD','BIIB','REGN','MRNA','CVX','XOM','COP','EOG',
           'NVDA','AMD','INTC','QCOM','AMAT','LRCX','KLAC','ASML','TSM',
           'NFLX','DIS','CMCSA','T','VZ','TMUS','SBUX','MCD','YUM',
           'CAT','DE','HON','ETN','EMR','GE','RTX','LMT','NOC','GD',
           'NOW','CRM','MSFT','ORCL','ADBE','SNOW','DDOG','PANW','CRWD',
           'COIN','PLTR','UBER','LYFT','DASH','ABNB','RBLX','HOOD'}

def _spread_pct(symbol: str) -> float:
    s = symbol.upper()
    if s in _TIER1: return 0.0001   # 0.01% — mega liquid ETFs
    if s in _TIER2: return 0.0003   # 0.03% — large-cap stocks
    return 0.0008                   # 0.08% — everything else (mid/small/volatile)

# ── Fill probability model ────────────────────────────────────────────────────
# Simulates realistic partial/rejected fills for paper trading realism
# Market orders: 95% chance of full fill, 4% partial (80% of qty), 1% reject
# We use market orders for simplicity — limit order modeling is backlog #future
import random as _random
def _simulate_fill(symbol: str, quantity: int) -> tuple[int, str]:
    """Returns (filled_qty, fill_status) — 'full' | 'partial' | 'rejected'"""
    roll = _random.random()
    if roll < 0.01:                          # 1% — order rejected
        return 0, 'rejected'
    elif roll < 0.05 and quantity > 10:      # 4% — partial fill (80% of qty)
        return max(1, int(quantity * 0.80)), 'partial'
    return quantity, 'full'                  # 95% — full fill
MIN_EDGE_MULTIPLIER = 2.0               # trade must profit at least 2x estimated costs


def initialize_portfolio(starting_cash=25000.00):
    portfolio = {
        'cash': starting_cash,
        'starting_cash': starting_cash,
        'positions': {},
        'orders': [],
        'trade_log': [],
        'total_fees_paid': 0.0,
        'total_spread_cost': 0.0,
        'created': datetime.now().isoformat(),
        'last_updated': datetime.now().isoformat()
    }
    save_portfolio(portfolio)
    print(f"✅ Paper portfolio initialized with ${starting_cash:,.2f}")
    return portfolio


def load_portfolio():
    if os.path.exists(PAPER_PORTFOLIO_FILE):
        with open(PAPER_PORTFOLIO_FILE, 'r') as f:
            p = json.load(f)
        # Back-fill fee fields if loading older portfolio
        p.setdefault('total_fees_paid', 0.0)
        p.setdefault('total_spread_cost', 0.0)
        return p
    return initialize_portfolio()


def save_portfolio(portfolio):
    portfolio['last_updated'] = datetime.now().isoformat()
    with open(PAPER_PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2)


def get_live_price(client, symbol):
    try:
        resp = client.get_quote(symbol)
        data = resp.json()
        asset = data.get(symbol, {})
        quote = asset.get('quote', {})
        regular = asset.get('regular', {})
        extended = asset.get('extended', {})

        # Try regular market price first, then quote, then extended
        candidates = [
            regular.get('regularMarketLastPrice'),
            quote.get('lastPrice'),
            quote.get('mark'),
            extended.get('lastPrice'),
        ]
        last = next((float(v) for v in candidates if v and float(v) > 0), 0)

        bid = quote.get('bidPrice', 0)
        ask = quote.get('askPrice', 0)
        mid = round((bid + ask) / 2, 4) if bid and ask else last

        return {'bid': bid, 'ask': ask, 'last': last, 'mid': mid}
    except Exception as e:
        print(f"Error getting price for {symbol}: {e}")
        return None


# ── Fee Estimator ─────────────────────────────────────────────────────────────

def estimate_fees(trade_value: float, contracts: int = 0,
                  symbol: str = '') -> dict:
    """
    Estimates total trading cost for a given trade.
    Returns breakdown of spread cost, commission, and total.
    """
    sp_pct  = _spread_pct(symbol) if symbol else SPREAD_COST_PCT
    spread  = round(trade_value * sp_pct, 4)
    commission = round(contracts * OPTIONS_COMMISSION_PER_CONTRACT, 4)
    total   = round(spread + commission, 4)
    min_profit = round(total * MIN_EDGE_MULTIPLIER, 4)
    return {
        'spread_cost':    spread,
        'commission':     commission,
        'total_fees':     total,
        'min_profit_needed': min_profit,
    }


def check_min_edge(trade_value: float, expected_profit: float, contracts: int = 0) -> tuple[bool, dict]:
    """
    Returns (passes, fee_detail).
    Passes if expected_profit >= 2x estimated fees.
    Use before entering a trade to filter out low-edge setups.
    """
    fees = estimate_fees(trade_value, contracts)
    passes = expected_profit >= fees['min_profit_needed']
    return passes, fees


# ── Buy ───────────────────────────────────────────────────────────────────────

def paper_buy(client, symbol, quantity, contracts: int = 0):
    """
    Executes a paper buy.
    contracts: number of options contracts (0 for equity trades).
    """
    portfolio  = load_portfolio()
    price_data = get_live_price(client, symbol)

    if not price_data:
        print(f"❌ Could not get price for {symbol}")
        return False

    fill_price  = price_data['ask'] if price_data['ask'] > 0 else price_data['last']
    trade_value = fill_price * quantity
    # Simulate fill — partial fills and rare rejections
    filled_qty, fill_status = _simulate_fill(symbol, quantity)
    if fill_status == 'rejected':
        print(f"⚠️  PAPER ORDER REJECTED: {symbol} x{quantity} — simulating Schwab rejection")
        return False
    if fill_status == 'partial':
        print(f"⚠️  PAPER PARTIAL FILL: {symbol} {filled_qty}/{quantity} shares filled")
        quantity = filled_qty

    trade_value = fill_price * quantity
    fees        = estimate_fees(trade_value, contracts, symbol)
    total_cost  = round(trade_value + fees['total_fees'], 4)

    if total_cost > portfolio['cash']:
        print(f"❌ Insufficient cash. Need ${total_cost:,.2f} (incl. fees), have ${portfolio['cash']:,.2f}")
        return False

    portfolio['cash'] -= total_cost
    portfolio['total_fees_paid']   = round(portfolio.get('total_fees_paid', 0)   + fees['commission'], 4)
    portfolio['total_spread_cost'] = round(portfolio.get('total_spread_cost', 0) + fees['spread_cost'], 4)

    if symbol in portfolio['positions']:
        existing  = portfolio['positions'][symbol]
        total_qty = existing['quantity'] + quantity
        avg_price = ((existing['avg_price'] * existing['quantity']) + (fill_price * quantity)) / total_qty
        portfolio['positions'][symbol] = {
            'quantity':   total_qty,
            'avg_price':  round(avg_price, 4),
            'cost_basis': round(avg_price * total_qty, 2),
            'source':     existing.get('source', 'standard'),  # preserve source tag
        }
    else:
        portfolio['positions'][symbol] = {
            'quantity':   quantity,
            'avg_price':  fill_price,
            'cost_basis': round(fill_price * quantity, 2),
            'source':     'standard',  # overridden by caller if politician/wsb
        }

    trade = {
        'type':            'BUY',
        'symbol':          symbol,
        'quantity':        quantity,
        'fill_status':     fill_status,
        'fill_price':      fill_price,
        'trade_value':     round(trade_value, 2),
        'spread_cost':     fees['spread_cost'],
        'commission':      fees['commission'],
        'total_fees':      fees['total_fees'],
        'total_cost':      round(total_cost, 2),
        'timestamp':       datetime.now().isoformat(),
        'cash_remaining':  round(portfolio['cash'], 2),
    }
    portfolio['trade_log'].append(trade)
    save_portfolio(portfolio)

    print(f"✅ PAPER BUY:  {quantity} {symbol} @ ${fill_price:.2f} = ${trade_value:,.2f}")
    print(f"   Fees:  spread ${fees['spread_cost']:.2f}  +  commission ${fees['commission']:.2f}  =  ${fees['total_fees']:.2f}")
    print(f"   Total cost: ${total_cost:,.2f}  |  Cash remaining: ${portfolio['cash']:,.2f}")
    return True


# ── Sell ──────────────────────────────────────────────────────────────────────

def paper_sell(client, symbol, quantity, contracts: int = 0):
    """
    Executes a paper sell.
    contracts: number of options contracts (0 for equity trades).
    """
    portfolio = load_portfolio()

    if symbol not in portfolio['positions']:
        print(f"❌ No position in {symbol}")
        return False

    position = portfolio['positions'][symbol]
    if quantity > position['quantity']:
        print(f"❌ Can't sell {quantity} shares, only have {position['quantity']}")
        return False

    price_data = get_live_price(client, symbol)
    if not price_data:
        print(f"❌ Could not get price for {symbol}")
        return False

    fill_price     = price_data['bid'] if price_data['bid'] > 0 else price_data['last']
    trade_value    = fill_price * quantity
    fees           = estimate_fees(trade_value, contracts, symbol)
    total_proceeds = round(trade_value - fees['total_fees'], 4)
    profit_loss    = round((fill_price - position['avg_price']) * quantity - fees['total_fees'], 4)

    portfolio['cash'] += total_proceeds
    portfolio['total_fees_paid']   = round(portfolio.get('total_fees_paid', 0)   + fees['commission'], 4)
    portfolio['total_spread_cost'] = round(portfolio.get('total_spread_cost', 0) + fees['spread_cost'], 4)

    if quantity == position['quantity']:
        del portfolio['positions'][symbol]
    else:
        portfolio['positions'][symbol]['quantity']   -= quantity
        portfolio['positions'][symbol]['cost_basis']  = round(
            portfolio['positions'][symbol]['avg_price'] * portfolio['positions'][symbol]['quantity'], 2
        )

    trade = {
        'type':            'SELL',
        'symbol':          symbol,
        'quantity':        quantity,
        'fill_price':      fill_price,
        'trade_value':     round(trade_value, 2),
        'spread_cost':     fees['spread_cost'],
        'commission':      fees['commission'],
        'total_fees':      fees['total_fees'],
        'total_proceeds':  round(total_proceeds, 2),
        'profit_loss':     profit_loss,
        'timestamp':       datetime.now().isoformat(),
        'cash_remaining':  round(portfolio['cash'], 2),
    }
    portfolio['trade_log'].append(trade)
    save_portfolio(portfolio)

    pl_emoji = "🟢" if profit_loss >= 0 else "🔴"
    print(f"✅ PAPER SELL: {quantity} {symbol} @ ${fill_price:.2f} = ${trade_value:,.2f}")
    print(f"   Fees:  spread ${fees['spread_cost']:.2f}  +  commission ${fees['commission']:.2f}  =  ${fees['total_fees']:.2f}")
    print(f"   Net proceeds: ${total_proceeds:,.2f}")
    print(f"   P&L (after fees): {pl_emoji} ${profit_loss:,.2f}  |  Cash remaining: ${portfolio['cash']:,.2f}")
    return True


# ── Portfolio Summary ─────────────────────────────────────────────────────────

def print_portfolio_summary(client):
    portfolio = load_portfolio()

    print("\n" + "="*57)
    print("📄 PAPER TRADING PORTFOLIO SUMMARY")
    print("="*57)
    print(f"  Cash:             ${portfolio['cash']:>12,.2f}")

    total_market_value = 0

    if portfolio['positions']:
        print(f"\n  {'SYMBOL':<10} {'QTY':<8} {'AVG':<10} {'PRICE':<10} {'VALUE':<12} {'P&L'}")
        print("  " + "-"*55)

        for symbol, pos in portfolio['positions'].items():
            price_data    = get_live_price(client, symbol)
            current_price = price_data['last'] if price_data else pos['avg_price']
            market_value  = current_price * pos['quantity']
            pl            = (current_price - pos['avg_price']) * pos['quantity']
            pl_str        = f"${pl:+,.2f}"
            total_market_value += market_value

            print(f"  {symbol:<10} {pos['quantity']:<8} ${pos['avg_price']:<9.2f} "
                  f"${current_price:<9.2f} ${market_value:<11,.2f} {pl_str}")
    else:
        print("\n  No open positions")

    total_value   = portfolio['cash'] + total_market_value
    total_pl      = total_value - portfolio['starting_cash']
    total_pl_pct  = (total_pl / portfolio['starting_cash']) * 100
    total_fees    = portfolio.get('total_fees_paid', 0)
    total_spread  = portfolio.get('total_spread_cost', 0)
    total_cost    = round(total_fees + total_spread, 4)

    print("\n" + "-"*57)
    print(f"  Positions Value:    ${total_market_value:>12,.2f}")
    print(f"  Total Value:        ${total_value:>12,.2f}")
    pl_emoji = "🟢" if total_pl >= 0 else "🔴"
    print(f"  Total P&L:       {pl_emoji}  ${total_pl:>+12,.2f} ({total_pl_pct:+.2f}%)")
    print(f"\n  ── Fee Breakdown ──────────────────────────────")
    print(f"  Spread costs paid:  ${total_spread:>12,.4f}")
    print(f"  Commissions paid:   ${total_fees:>12,.4f}")
    print(f"  Total friction:     ${total_cost:>12,.4f}")
    print(f"  P&L net of fees:  {pl_emoji}  ${total_pl:>+12,.2f}")
    print(f"\n  Trades executed:    {len(portfolio['trade_log'])}")
    print(f"  Last updated:       {portfolio['last_updated'][:19]}")
    print("="*57)


if __name__ == '__main__':
    client = authenticate()
    paper = True  # assume paper mode

    if not os.path.exists(PAPER_PORTFOLIO_FILE):
        initialize_portfolio(starting_cash=25000.00)

    print_portfolio_summary(client)

    print("\n🧪 Running test trades...")
    paper_buy(client, 'SPY', 5)
    paper_buy(client, 'QQQ', 3)
    print_portfolio_summary(client)