__author__ = 'Mark'

from Strategies import *
from utils import backtest

#strategies = [MomentumStrategy, BuyAndHoldStrategy]
#strategy_test(strategies, tickers)
tickers = ['AAPL', 'QQQ', 'KMI', 'VZ', 'DD', 'VOD', 'CTL']
ms = MomentumStrategy(10000, tickers)
bs = BuyAndHoldStrategy(10000, tickers)
ts = TSEBuyAndHoldStrategy(10000, tickers)
print(backtest(ms, correct=False))
print(backtest(bs, correct=False))
print(backtest(ts, correct=False))

