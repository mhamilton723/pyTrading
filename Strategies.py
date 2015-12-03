from collections import deque
from Portfolio import Portfolio


class Strategy(object):

    def __init__(self, balance, log=False):
        self.portfolio = Portfolio(balance)
        self._log = log
        self._today_data = None

    def observe_data(self, data_stream):
        for i in range(data_stream.shape[1]):
            self._today_data = data_stream.iloc[:, i, :]
            self.observe_datum(data_stream.iloc[:, i, :])
            self.act()

    def observe_datum(self, datum):
        raise NotImplementedError

    def log(self, string):
        if self._log:
            print(string)

    def liquidate(self, datum):
        for ticker in self.portfolio.tickers():
            if self.portfolio.owns(ticker):
                self.portfolio.sell_max(ticker, float(datum['Adj Close'][ticker]))

    def act(self):
        raise NotImplementedError

    def value(self, datum=None, correct=True):
        if not datum:
            datum = self._today_data
        return self.portfolio.value(datum, correct)


class SingleStockStrategy(Strategy):
    def __init__(self, balance, ticker, log=False):
        super(SingleStockStrategy, self).__init__(balance, log=log)
        self.ticker = ticker


class MultiStockStrategy(Strategy):
    def __init__(self, balance, tickers, log=False):
        super(MultiStockStrategy, self).__init__(balance, log=log)

        if type(tickers) is str:
            tickers = [tickers] #make it polymorphic in the case of one ticker

        self.tickers = tickers


class WeightedMultiStockStrategy(MultiStockStrategy):

    def __init__(self, balance, tickers, weights='uniform', log=False):
        super(WeightedMultiStockStrategy, self).__init__(balance, tickers, log=log)
        if weights == 'uniform':
            self.weights = {ticker: 1. / len(self.tickers) for ticker in self.tickers}
        else:
            self.weights = {ticker: weights[i] / float(sum(weights)) for i, ticker in enumerate(self.tickers)}



class BuyAndHoldStrategy(WeightedMultiStockStrategy):
    def __init__(self, balance, tickers, weights='uniform', log=False):
        super(BuyAndHoldStrategy, self).__init__(balance, tickers, weights=weights, log=log)

    def observe_data(self, data_stream, pandas=True):
        for ticker in self.tickers:
            self.portfolio.buy_max(ticker,
                                   float(data_stream.iloc[:, 0, :]['Adj Close'][ticker]),
                                   self.weights[ticker])
            self._today_data = data_stream.iloc[:, -1, :]

    def __str__(self):
        return "Buy and Hold Strategy"


class MomentumStrategy(WeightedMultiStockStrategy):
    def __init__(self, balance, tickers, window=50, weights='uniform', log=False):
        super(MomentumStrategy, self).__init__(balance, tickers, weights=weights, log=log)
        self.window = window
        self._cur_prices = {}
        self._bought_prices = {}
        self._last_n_prices = deque(maxlen=window)

    def observe_datum(self, datum, **kwargs):
        for ticker in self.tickers:
            self._cur_prices[ticker] = datum['Adj Close'][ticker]
        self._last_n_prices.append(datum['Adj Close'][self.tickers])

    def act(self):
        if len(self._last_n_prices) == self.window:
            moving_average = sum(self._last_n_prices) / float(self.window)
            for ticker in self.tickers:
                if not self.portfolio.owns(ticker):
                    if moving_average[ticker] < self._cur_prices[ticker]:
                        self._bought_prices[ticker] = self._cur_prices[ticker]
                        self.portfolio.buy_max(ticker, self._cur_prices[ticker], self.weights[ticker])
                        self.log("Bought stock at " + str(self._cur_prices[ticker]))
                else:
                    if moving_average[ticker] > self._cur_prices[ticker]:
                        self.portfolio.sell_max(ticker, self._cur_prices[ticker])
                        self.log("Sold stock at " + str(self._cur_prices[ticker]) + " for profit of " +
                                 str(self._cur_prices[ticker] - self._bought_prices[ticker]) + " per share")
                        self._bought_prices[ticker] = None

    def __str__(self):
        return "Momentum Strategy"








