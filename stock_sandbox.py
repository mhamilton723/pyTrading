from __future__ import print_function
import pandas.io.data as web
from datetime import datetime
from utils import backtest
import numpy as np
from Strategies import MomentumStrategy, BuyAndHoldStrategy
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt


import pandas

import statsmodels.api as sm

import math
import random
from utils import load_s_and_p_data




'''
df = get_data('AAPL')
cf_cycles, cf_trend = sm.tsa.filters.cffilter(df)

fig = plt.figure(figsize=(18,8))
ax1 = fig.add_subplot(311)
df.plot()
cf_trend.plot()
ax2 = fig.add_subplot(312)
fig = sm.graphics.tsa.plot_pacf(df, lags=50, ax=ax2)
arma_mod20 = sm.tsa.ARMA(df, (2,0)).fit()
arma_mod30 = sm.tsa.ARMA(df, (3,0)).fit()
ax = fig.add_subplot(313)
ax = df.ix[:].plot(ax=ax)
fig = arma_mod30.plot_predict(int(len(df)*3./4), len(df)-1, dynamic=True, ax=ax, plot_insample=False)
plt.show()
'''


def auto_regression_multi(df, window=3, pandas=True):
    x_size = len(df) - window - 1
    n_features = len(df.columns)
    X = np.empty((x_size, window * n_features))
    y = np.empty((x_size, n_features))
    for i in range(x_size):
        for f_i in range(n_features):
            if pandas:
                X[i, window * f_i:window * (f_i + 1)] = df.iloc[i:i + window, f_i]
                y[i, f_i] = df.iloc[i + window, f_i]
            else:
                X[i, window * f_i:window * (f_i + 1)] = df[i:i + window, f_i]
                y[i, f_i] = df[i + window, f_i]
    return X, y


def auto_regression_format(df, window=3, pandas=True):
    x_size = len(df) - window - 1
    X = np.empty((x_size, window))
    y = np.empty((x_size,))
    for i in range(x_size):
        if pandas:
            X[i, :] = df.iloc[i:i + window]
            y[i] = df.iloc[i + window]
        else:
            X[i, :] = df[i:i + window]
            y[i] = df[i + window]
    return X, y


def forecast(model, X_train, window=10, n_points=300, pandas=True, percent_noise = .002):
    X, y = auto_regression_multi(X_train, window=window, pandas=pandas)
    model.fit(X, y)
    output = []
    x_init = list(X[-1])
    n_features = len(x_init) / window
    for i in range(n_points):
        y_pred = model.predict([x_init])[0]
        y_pred = [y + y*(.5-random.random())*percent_noise for y in y_pred]
        output.append(y_pred)

        x_new = []
        for f_i in range(n_features):
            for j in range(1, window):
                start = f_i * window
                x_new.append(x_init[start + j])
            x_new.append(y_pred[f_i])
        x_init = x_new

    return output


def test_train_plot(model, train_data, test_data, window=10, pandas=True, n_estimators=10):
    for k in range(n_estimators):
        output = forecast(model, train_data, window=window, n_points=len(test_data), pandas=pandas,percent_noise=.05)
        output = np.array(output)
        best_shape = (output.shape[1],1) if output.shape[1] < 4 else (math.ceil(output.shape[1]**.5),math.ceil(output.shape[1]**.5))
        for i in range(output.shape[1]):
            plt.subplot(best_shape[0], best_shape[1], i+1)
            predicted = output[:, i]
            actual = test_data.iloc[:, i]
            plt.plot(predicted, color='r')
            plt.plot(actual, color='b')
            plt.title(train_data.columns[i])
    plt.show()


'''
training_samples = 10000
testing_samples = 300
def regression_target(x):
    return math.sin(x/10.) + x**.5 + random.random()/100
train_data = [regression_target(x) for x in range(training_samples)]
test_data = [regression_target(x) for x in range(training_samples, training_samples+testing_samples)]
#model = linear_model.LinearRegression()
model = KernelRidge(kernel='poly')
test_train_plot(model, train_data, test_data, window=20)


df = get_data(['AAPL', 'QQQ','VZ','NKE','KMI'])
train_data = df.iloc[0:int(len(df) / 1.5), :]
test_data = df.iloc[int(len(df) / 1.5):len(df), :]
dim_reducer = PCA(n_components=100)
regressor = linear_model.LinearRegression()
model = Pipeline([('PCA',dim_reducer),('regressor', regressor)])
test_train_plot(model, train_data, test_data, window=20)
'''




#strategies = [MomentumStrategy, BuyAndHoldStrategy]
#strategy_test(strategies, tickers)
tickers = ['AAPL', 'QQQ', 'KMI', 'VZ', 'DD', 'VOD', 'CTL']
ms = MomentumStrategy(10000, tickers)
bs = BuyAndHoldStrategy(10000, tickers)
print(backtest(ms, correct=False))
print(backtest(bs, correct=False))

#data = load_s_and_p_data()

#print(np.arange(1, 6))
#print(np.diag(np.arange(1, 6), 1))
#print(np.diag(np.arange(1, 6), 2))
#data.to_csv('stock_sandbox/data/sp500_data.pkl"')

# print(data.head())
# backtest_multi_stock(tickers)
