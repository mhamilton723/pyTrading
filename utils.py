import pandas.io.data as web
from datetime import datetime
import pickle
import os
import random
import numpy as np
import pandas as pd

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

def strategy_test(strategies, tickers, start="2014-1-1", end="2015-11-02", starting_capital=1000):
    for strategy_object in strategies:
        for ticker in tickers:
            strategy = strategy_object(starting_capital, ticker)
            momentum_result = round(backtest(strategy,start=start, end=end), 2)
            print("Percent return for " + str(strategy) + " for stock " + ticker + ": %" + str(momentum_result))
	
def get_data(tickers, start="2014-1-1", end="2015-11-02"):
    start_time = datetime.strptime(start, "%Y-%m-%d")
    end_time = datetime.strptime(end, "%Y-%m-%d")
    df = web.DataReader(tickers, 'yahoo', start_time, end_time)
    df = df['Adj Close']
    #df = df.diff()
    #df = df.iloc[1:len(df),:]

    return df

def load_s_and_p_data(start="2009-1-1", end="2015-11-02",
                          ticker_names = "~/machine_learning/stock_sandbox/s_and_p_500_names.csv", 
						  tickers = None, clean=True):

    if not tickers:
		s_and_p = pd.read_csv(ticker_names)
		tickers = list(s_and_p['Ticker'])
    data = get_data(tickers, start=start, end=end)
    if clean:
        data = data.dropna(axis=1)

    return data


def window_dataset(data, n_prev=1):
    """
	data should be pd.DataFrame()
	"""
    docX, docY = [], []
    for i in range(len(data) - n_prev):
        docX.append(data.iloc[i:i + n_prev].as_matrix())
        docY.append(data.iloc[i + n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY



def train_test_split(df, test_size=0.1, n_prev=1):
    """
	This just splits data to training and testing parts
	"""

    ntrn = round(len(df) * (1 - test_size))

    X_train, y_train = window_dataset(df.iloc[0:ntrn], n_prev=n_prev)
    X_test, y_test = window_dataset(df.iloc[ntrn:], n_prev=n_prev)

    return (X_train, y_train), (X_test, y_test)


def forecast(model, seed, n_points=300, percent_noise=.002):
    output = np.empty((n_points, seed.shape[1]))
    values = np.empty((n_points, seed.shape[0], seed.shape[1]))
    values[0, :] = seed

    for i in range(n_points):
        y_pred = model.predict(values[[i], :])[0]
        y_pred = np.array([[y + y * (.5 - random.random()) * percent_noise for y in y_pred]])

        output[i, :] = y_pred
        if i < n_points - 1:
            if len(seed.shape) > 2:
                print(values[i + 1, :].shape, values[i, 1:, :].shape, y_pred.shape)
                values[i + 1, :] = np.hstack((values[i, 1:, :], y_pred))
            else:
                values[i + 1, :] = y_pred

    return output
