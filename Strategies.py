from collections import deque
from Portfolio import Portfolio


class Strategy(object):
    def __init__(self, balance, log=False):
        self.portfolio = Portfolio(balance)
        self._log = log


    def observe_data(self, data_stream, pandas=True):
        if pandas:
            for i, data_point in data_stream.iterrows():
                self.observe_datum(data_point)
                self.act()
        else:
            for data_point in data_stream:
                self.observe_datum(data_point)
                self.act()

        self.liquidate(data_stream.tail(1))

    def observe_datum(self, datum):
        raise NotImplementedError

    def log(self, string):
        if self._log:
            print(string)

    def liquidate(self, datum):
        raise NotImplementedError

    def act(self):
        raise NotImplementedError



class SingleStockStrategy(Strategy):
    def __init__(self, balance, ticker, log=False):
        super(SingleStockStrategy, self).__init__(balance, log=log)
        self.ticker = ticker

    def liquidate(self, datum):
        if self.portfolio.owns(self.ticker):
            self.portfolio.sell_max(self.ticker, float(datum['Adj Close']))


class BuyAndHoldStrategy(SingleStockStrategy):
    def __init__(self, balance, ticker, log=False):
        super(BuyAndHoldStrategy, self).__init__(balance, ticker, log=log)

    def observe_data(self, data_stream, pandas=True):
        if pandas:
            self.portfolio.buy_max(self.ticker, float(data_stream.head(1)['Adj Close']))
        else:
            self.portfolio.buy_max(self.ticker, data_stream[0]['Adj Close'])

        self.liquidate(data_stream.tail(1))

    def __str__(self):
        return "Buy and Hold Strategy"

class MomentumStrategy(SingleStockStrategy):
    def __init__(self, balance, ticker, window=50, log=False):
        super(MomentumStrategy, self).__init__(balance, ticker, log=log)
        self.window = window
        self._own_stock = False
        self._price = None
        self._bought_price = None
        self._last_n_values = deque(maxlen=window)

    def observe_datum(self, datum, **kwargs):
        self._price = datum['Adj Close']
        self._last_n_values.append(datum['Adj Close'])

    def act(self):
        if len(self._last_n_values) == self.window:
            moving_average = sum(self._last_n_values) / float(self.window)

            if not self._own_stock:
                if moving_average < self._price:
                    self._own_stock = True
                    self._bought_price = self._price
                    self.portfolio.buy_max(self.ticker, self._price)
                    self.log("Bought stock at " + str(self._price))
            else:
                if moving_average > self._price:
                    self._own_stock = False
                    self.portfolio.sell_max(self.ticker, self._price)
                    self.log("Sold stock at " + str(self._price) + " for profit of " +
                             str(self._price - self._bought_price) + " per share")
                    self._bought_price = None

    def __str__(self):
        return "Momentum Strategy"


