from collections import deque
from Portfolio import Portfolio
from TimeSeriesEstimator import TimeSeriesRegressor
from sklearn.linear_model import LinearRegression
import itertools
import numpy as np
import random
from utils import cache, load_s_and_p_data,s_and_p_names

class Strategy(object):

    def __init__(self, balance, log=False, commission=.0002, flat_rate=8):
        self.portfolio = Portfolio(balance, commission, flat_rate)
        self._log = log
        self._today_data = None
        self.day = 0

    def run(self, data_stream):
        for i in range(data_stream.shape[1]):
            self._today_data = data_stream.iloc[:, i, :]
            self.observe_datum(data_stream.iloc[:, i, :])
            self.act()
            self.day += 1

    def observe_datum(self, datum):
        raise NotImplementedError

    def log(self, string):
        if self._log:
            print(string)

    def liquidate(self):
        for ticker in self.portfolio.tickers():
            self.sell_max(ticker)

    def act(self):
        raise NotImplementedError

    def price(self, tickers):
        return self._today_data['Adj Close'][tickers]

    def value(self, correct=True):
        datum = self._today_data
        return self.portfolio.value(datum, correct)

    def sell(self, ticker, shares=1):
        self.portfolio.sell(ticker, self.price(ticker), shares)

    def sell_max(self, ticker):
        self.portfolio.sell_max(ticker, self.price(ticker))

    def buy(self, ticker, shares=1):
        self.portfolio.buy(ticker, self.price(ticker), shares)

    def buy_max(self, ticker, weight=1.):
        self.portfolio.buy_max(ticker, self.price(ticker), weight)

    def batch_buy(self, tickers, weights):
        if type(weights) is dict:
            weights = [weights[ticker] for ticker in tickers]
        if sum(weights) > 1:
            weights = [w/sum(weights) for w in weights]

        self.portfolio.batch_buy(tickers, self.price(tickers), weights)


class SingleStockStrategy(Strategy):
    def __init__(self, balance, ticker,
                 log=False, commission=.0002, flat_rate=8):
        super(SingleStockStrategy, self).__init__(balance,
                                                  log=log, commission=commission, flat_rate=flat_rate)
        self.ticker = ticker


class MultiStockStrategy(Strategy):
    def __init__(self, balance, tickers,
                 log=False, commission=.0002, flat_rate=8):
        super(MultiStockStrategy, self).__init__(balance,
                                                 log=log, commission=commission, flat_rate=flat_rate)

        if type(tickers) is str:
            tickers = [tickers]  # make it polymorphic in the case of one ticker

        self.tickers = tickers


class WeightedMultiStockStrategy(MultiStockStrategy):

    def __init__(self, balance, tickers, weights='uniform',
                 log=False, commission=.0002, flat_rate=8):
        super(WeightedMultiStockStrategy, self).__init__(balance, tickers,
                                                         log=log, commission=commission, flat_rate=flat_rate)
        if weights == 'uniform':
            self.weights = {ticker: 1. / len(self.tickers) for ticker in self.tickers}
        elif type(weights) is list:
            self.weights = {ticker: weights[i] / float(sum(weights)) for i, ticker in enumerate(self.tickers)}
        else:
            raise ValueError("Must give 'uniform' or a list of weights")


class BuyAndHoldStrategy(WeightedMultiStockStrategy):
    def __init__(self, balance, tickers, weights='uniform', wait=0,
                 log=False, commission=.0002, flat_rate=8, fast=True):
        super(BuyAndHoldStrategy, self).__init__(balance, tickers,
                                                 weights=weights, log=log, commission=commission, flat_rate=flat_rate)
        self.wait = wait
        self.fast = fast

    def run(self, data_stream):
        if self.fast:
            self._today_data = data_stream.iloc[:, self.wait, :]
            self.day = self.wait
            self.act()
            self._today_data = data_stream.iloc[:, -1, :]
            self.day = data_stream.shape[1] - 1
        else:
            super(BuyAndHoldStrategy, self).run(data_stream)

    def observe_datum(self, datum):
        pass

    def act(self):
        if self.day == self.wait:
            self.batch_buy(self.tickers, self.weights)

    def __str__(self):
        return "Buy and Hold Strategy"


class MomentumStrategy(WeightedMultiStockStrategy):
    def __init__(self, balance, tickers, window=50, weights='uniform',
                 log=False, commission=.0002, flat_rate=8):
        super(MomentumStrategy, self).__init__(balance, tickers,
                                               weights=weights, log=log, commission=commission, flat_rate=flat_rate)
        self.window = window
        self._bought_prices = {}
        self._last_n_prices = deque(maxlen=window)

    def observe_datum(self, datum, **kwargs):
        self._last_n_prices.append(datum['Adj Close'][self.tickers])

    def act(self):
        if len(self._last_n_prices) == self.window:  # check if we have seen enough data
            moving_average = sum(self._last_n_prices) / float(self.window)
            for ticker in self.tickers:
                if not self.portfolio.owns(ticker):
                    if moving_average[ticker] < self.price(ticker):
                        self._bought_prices[ticker] = self.price(ticker)
                        self.buy_max(ticker, self.weights[ticker])
                        self.log("Bought stock at " + str(self.price(ticker)))
                else:
                    if moving_average[ticker] > self.price(ticker):
                        self.sell_max(ticker)
                        self.log("Sold stock at " + str(self.price(ticker)) + " for profit of " +
                                 str(self.price(ticker) - self._bought_prices[ticker]) + " per share")
                        self._bought_prices[ticker] = None

    def __str__(self):
        return "Momentum Strategy"


class InformedBuyAndHoldStrategy(MultiStockStrategy):

    def __init__(self, balance, tickers, wait=100,
                 log=False, commission=.0002, flat_rate=8):
        super(InformedBuyAndHoldStrategy, self).__init__(balance, tickers,
                                          log=log, commission=commission, flat_rate=flat_rate)

        self.wait = wait
        self.observed_data = None
        self.names = None

    def observe_datum(self, datum, **kwargs):
        if self.observed_data is None:
            self.observed_data = np.zeros((1, len(datum)))
            self.observed_data[0, :] = np.array(datum['Adj Close'])
            self.names = datum.index.values
        else:
            self.observed_data = np.vstack((self.observed_data, np.array(datum['Adj Close'])))  # TODO make more general

    def act(self):
        if self.wait == self.day:
            tickers, weights = self.choose_stocks()
            self.batch_buy(tickers, weights)

    def choose_stocks(self):
        raise NotImplementedError

class TSEBuyAndHoldStrategy(InformedBuyAndHoldStrategy):
    def __init__(self, balance, tickers=None, base_model=LinearRegression(),
                 n_prev=2, wait=100, steps_ahead=100, k=5, envelope='proportional',
                 log=False, commission=.0002, flat_rate=8):

        if tickers is None:
            tickers = s_and_p_names()

        super(TSEBuyAndHoldStrategy, self).__init__(balance, tickers,
                                          log=log, commission=commission, flat_rate=flat_rate)
        self.model = TimeSeriesRegressor(base_model, n_ahead=1, n_prev=n_prev)
        self.day = 0
        self.wait = wait
        self.steps_ahead = steps_ahead
        self.observed_data = None
        self.k = k
        self.envelope = envelope


    def choose_stocks(self):
        self.model.fit(self.observed_data)
        fc = self.model.forecast(self.observed_data, self.steps_ahead)
        changes = np.array([fc[-1, i] - fc[0, i] for i in range(fc.shape[1])])
        top_k = changes.argsort()[::-1]
        top_tickers = self.names[top_k]
        if self.envelope == 'proportional':
            top_weights = changes[top_k]
        elif self.envelope == 'log_proportional':
            top_weights = np.log(changes[top_k])
        elif self.envelope == 'uniform':
            top_weights = np.ones((self.k))
        else:
            raise ValueError("Chose a proper strategy name")

        top_weights = np.array(map(lambda w: max(0, w), top_weights))
        top_weights = top_weights / float(sum(top_weights))
        return top_tickers, top_weights

    def __str__(self):
        return "TSE Buy and Hold Strategy"


@cache('data/buy_and_hold_spread_cache.pkl')
def buy_and_hold_spread(k=5, wait=100, start="2014-1-1", end="2015-11-02", iterations=100):
    sp500_names = s_and_p_names(start, end)
    out = []
    if iterations == 'full':
        for tickers in itertools.combinations(sp500_names, k):
            tickers = list(tickers)
            bs = BuyAndHoldStrategy(10000, tickers, wait=wait)
            out.append(backtest(bs, start=start, end=end, correct=False) )
    else:
        for i in range(iterations):
            tickers = random.sample(sp500_names, k)
            bs = BuyAndHoldStrategy(10000, tickers, wait=wait)
            out.append(backtest(bs, start=start, end=end, correct=False) )

    return out


def backtest(strategy, start="2014-1-1", end="2015-11-02", log=False, correct=True):
    """
    :param start: starting date in %Y-%m-%d
    :param end: ending date in %Y-%m-%d
    :param log: flag to turn on logging
    :return: return relative to first stock purchase
    """
    #df = get_data(strategy.tickers, start, end)
    df = load_s_and_p_data(start=start, end=end, only_close=False)
    if df.empty:
        raise ValueError("No stock data found")
    if log:
        print(df.describe())
        strategy._log = True
    starting_balance = strategy.portfolio.balance
    strategy.run(df)
    ending_value = strategy.value(correct=correct)
    if log:
        for transaction in strategy.portfolio.transactions:
            print(transaction)
        print(starting_balance, ending_value)

    return (ending_value - starting_balance) * 100. / starting_balance


def strategy_test(strategies, tickers, start="2014-1-1", end="2015-11-02", starting_capital=1000):
    for strategy_object in strategies:
        for ticker in tickers:
            strategy = strategy_object(starting_capital, ticker)
            backtest_result = round(backtest(strategy, start=start, end=end), 2)
            print("Percent return for " + str(strategy) + " for stock " + ticker + ": %" + str(backtest_result))
