__author__ = 'Mark'
import matplotlib.pyplot as plt
import numpy as np
from Strategies import buy_and_hold_spread, backtest, TSEBuyAndHoldStrategy
from utils import cache

@cache('data/buy_and_hold_spreads_cache.pkl')
def buy_and_hold_spreads(ks=range(1, 10), wait=100, start="2014-1-1", end="2015-11-02", iterations=300):
    return [buy_and_hold_spread(k, wait, start, end, iterations) for k in ks]

def tse_results(balance=10000, ks=range(1, 10), wait=100, envelope='uniform', start="2014-1-1", end="2015-11-02"):
    strategies = [TSEBuyAndHoldStrategy(balance=balance, k=k, wait=wait, envelope=envelope) for k in ks]
    return [backtest(strategy, start=start, end=end, correct=False) for strategy in strategies]

ks = range(1, 10)
bh_spreads = buy_and_hold_spreads(ks=ks)
tse_returns = tse_results(ks=ks)

probs = [sum(1 if tser > bhr else 0 for bhr in bhs) / float(len(bhs))
         for (tser, bhs) in zip(tse_returns, bh_spreads)]

plt.subplot(1, 2, 1)
plt.plot(ks, probs, 'r-')
plt.xlabel('k')
plt.ylabel('Percentage of Buy And Hold Strategies Beaten')
plt.subplot(1, 2, 2)
plt.boxplot(bh_spreads)
for i, tse_return in enumerate(tse_returns):
    x = np.random.normal(i + 1, 0.02, size=len(bh_spreads[i]))
    if i == 0:
        plt.plot(x, bh_spreads[i], 'bo', alpha=.3, label='BH Gains')
        plt.plot(i + 1, tse_return, 'ro', ms=10, label='TSE Gains')
    else:
        plt.plot(x, bh_spreads[i], 'bo', alpha=.3)
        plt.plot(i + 1, tse_return, 'ro', ms=10)
plt.xlabel('k')
plt.ylabel('Percentage Return')
plt.legend()
plt.gcf().set_size_inches(15,6)
plt.show()