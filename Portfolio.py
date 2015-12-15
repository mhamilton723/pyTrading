from collections import defaultdict


class Portfolio(object):
    def __init__(self, balance=0, commission=.0002, flat_rate=8, equity=None):
        self.flat_rate = flat_rate
        self.commission = commission
        self.balance = balance
        self.equity = defaultdict(int) if not equity else equity
        self.transactions = []

    def tickers(self):
        return self.equity.keys()

    def value(self, datum, correct=True):
        if correct:
            return sum(self.corrected_price(datum['Adj Close'][ticker],
                                            self.equity[ticker],
                                            buying=False) for ticker in self.tickers()) + self.balance
        else:
            return sum(datum['Adj Close'][ticker] * self.equity[ticker] for ticker in self.tickers()) + self.balance

    def owns(self, ticker):
        if ticker in self.equity:
            if self.equity[ticker] >= 1:
                return True
        return False

    def shares(self, ticker):
        if ticker in self.equity:
            return self.equity[ticker]
        else:
            return 0

    def corrected_price(self, price, shares, buying):
        if buying:
            return price * shares + (shares * price * self.commission) + self.flat_rate
        else:
            return price * shares - (shares * price * self.commission) - self.flat_rate

    def sell(self, ticker, price, shares=1):
        corrected_price = self.corrected_price(price, shares, buying=False)
        if self.equity[ticker] >= shares:
            self.equity[ticker] -= shares
            self.balance += corrected_price
            self.transactions.append((ticker, 'SELL', shares, price, corrected_price, self.balance))
        else:
            raise ValueError("Cannot sell what you do not own")

    def sell_max(self, ticker, price):
        if self.equity[ticker]:
            self.sell(ticker, price, self.equity[ticker])
        else:
            raise ValueError("Cannot sell what you do not own")

    def buy(self, ticker, price, shares=1):
        corrected_price = self.corrected_price(price, shares, buying=True)
        if self.balance >= corrected_price:
            self.equity[ticker] += shares
            self.balance -= corrected_price
            self.transactions.append((ticker, 'BUY', shares, price, corrected_price, self.balance))
        else:
            raise ValueError("Cannot afford to buy this stock")

    def buy_max(self, ticker, price, weight=1.):
        if price:
            corrected_price = price + (price * self.commission)
            if self.balance - self.flat_rate >= corrected_price:
                n_shares = (weight * self.balance - self.flat_rate) // (price + price * self.commission)
                self.buy(ticker, price, n_shares)
            else:
                raise ValueError("Cannot afford to buy this stock")
        else:
            raise ValueError('No price found, dataset may be missing values')

    def batch_buy(self, tickers, prices, weights):
        orders = []
        for price, weight in zip(prices, weights):
            n_shares = (weight * self.balance - self.flat_rate) // (price + price * self.commission)
            orders.append(n_shares)

        for ticker, price, n_shares in zip(tickers, prices, orders):
            self.buy(ticker, price, shares=n_shares)
