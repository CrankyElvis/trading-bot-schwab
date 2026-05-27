"""test_logging.py — exercise score_logger and chain_logger end-to-end."""
import os
os.environ['SCORE_LOG_DB'] = r'C:\trading-bot\data\scores_live.db'
os.environ['CHAIN_LOG_DB'] = r'C:\trading-bot\data\chains_live.db'

from auth import authenticate
from data_collector import collect_snapshot, get_vix, get_vix_history
from regime_engine import evaluate_regime
from flow_momentum import run_scoring_cycle, print_cycle_result, StockScore
from options_manager import evaluate_options

print("Authenticating...")
client, paper = authenticate()
print(f"Connected ({'PAPER' if paper else 'LIVE'})\n")

universe = ['SPY', 'AAPL', 'NVDA', 'MSFT', 'TSLA']
vixy = get_vix(client)
vix_hist = get_vix_history(client, days=30)
regime = evaluate_regime(vixy, vix_hist)

print(f"Collecting snapshot for {len(universe)} symbols...")
snapshot = collect_snapshot(client, universe)

print("\n=== Test 1: Score logging via run_scoring_cycle ===")
result = run_scoring_cycle(
    universe=universe, snapshot=snapshot,
    regime=regime.regime, portfolio_value=25000,
)
print_cycle_result(result)

print("\n=== Test 2: Chain logging via evaluate_options ===")
fake = StockScore(symbol='AAPL', total_score=0.90, qualifies=True,
                   direction='bullish', signals={})
portfolio = {'cash': 25000, 'positions': {}}
options_eval = evaluate_options(
    candidates=[fake], portfolio=portfolio,
    quotes=snapshot.get('quotes', {}),
    vixy=vixy, regime=regime.regime, client=client,
)
print(f"  CSPs returned:       {len(options_eval['csps'])}")
print(f"  Long calls returned: {len(options_eval['long_calls'])}")

print("\n=== Database state ===")
import score_logger, chain_logger
score_logger.summary_stats()
print()
chain_logger.summary_stats()