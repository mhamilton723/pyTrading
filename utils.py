import pandas.io.data as web
from datetime import datetime

def backtest(strategy, start="2014-1-1", end="2015-11-02", log=False):
    """
    :param start: starting date in %Y-%m-%d
    :param end: ending date in %Y-%m-%d
    :param log: flag to turn on logging
    :return: return relative to first stock purchase
    """
    start_time = datetime.strptime(start, "%Y-%m-%d")
    end_time = datetime.strptime(end, "%Y-%m-%d")
    df = web.DataReader(strategy.ticker, 'yahoo', start_time, end_time)

    if df.empty:
        raise ValueError("No stock data found")

    if log:
        print(df.describe())
        strategy._log = True

    starting_balance = strategy.portfolio.balance
    strategy.observe_data(df)
    ending_balance = strategy.portfolio.balance

    if log:
        for transaction in strategy.portfolio.transactions:
            print(transaction)
        print(starting_balance, ending_balance)

    return (ending_balance - starting_balance) * 100. / starting_balance

