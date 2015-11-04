from collections import defaultdict

class Portfolio(object):
    def __init__(self, balance=0, commission=.0002, flat_rate=8, equity=None):
        self.flat_rate = flat_rate
        self.commission = commission
        self.balance = balance
        self.equity = defaultdict(int) if not equity else equity
        self.transactions = []

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

    def sell(self, ticker, price, shares=1):
        corrected_price = price * shares - (shares * price * self.commission) - self.flat_rate
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
        corrected_price = price * shares + (shares * price * self.commission) + self.flat_rate
        if self.balance >= corrected_price:
            self.equity[ticker] += shares
            self.balance -= corrected_price
            self.transactions.append((ticker, 'BUY', shares, price, corrected_price, self.balance))
        else:
            raise ValueError("Cannot afford to buy this stock")

    def buy_max(self, ticker, price):
        corrected_price = price + (price * self.commission) + self.flat_rate
        if self.balance >= corrected_price:
            n_shares = int(self.balance / corrected_price)
            self.buy(ticker, price, n_shares)
        else:
            raise ValueError("Cannot afford to buy this stock")
