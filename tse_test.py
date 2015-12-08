import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from utils import datasets, time_series_split, safe_shape, mse
from TimeSeriesEstimator import TimeSeriesRegressor

X = datasets('sp500')
y = X
#X = datasets('synthetic')
#y = X

X_train, X_test = time_series_split(X)
y_train, y_test = time_series_split(y)


n_prevs = range(1, 10)
mses_train = np.empty((len(n_prevs), safe_shape(y_train, 1)))
mses_test = np.empty((len(n_prevs), safe_shape(y_train, 1)))
for i, n_prev in enumerate(n_prevs):
    model = linear_model.LinearRegression()
    time_model = TimeSeriesRegressor(model, n_prev=n_prev)
    time_model.fit(X_train, y_train)
    res_test = time_model.predict(X_test)
    res_train = time_model.predict(X_train)
    mses_train[i, :] = mse(res_train, y_train[n_prev:], multioutput='raw_values')
    mses_test[i, :] = mse(res_test, y_test[n_prev:], multioutput='raw_values')


plt.boxplot(np.transpose(mses_test))
plt.yscale('log')


plot_train = False
if plot_train: res_plot = res_train; y_plot = y_train
else: res_plot = res_test; y_plot = y_test
low = 0
high = 50#len(res_test)-n_prev

plot_res_vs_actual = False
plot_scatter = False
if plot_res_vs_actual:
    for dim in range(min(safe_shape(res_plot, 1), 4)):
        plt.subplot(2, 2, dim + 1)

        if safe_shape(res_test, 1) == 1:
            plt.plot(res_plot[low:high], label='Predicted')
            plt.plot(y_plot[n_prev + low : n_prev + high], label='Actual')
        else:
            plt.plot(res_plot[low:high, dim], label='Predicted')
            plt.plot(y_plot[n_prev + low : n_prev + high, dim], label='Actual')
        plt.legend(loc='lower left')

if plot_scatter:
    for dim in range(min(safe_shape(res_plot, 1), 4)):
        plt.subplot(2, 2, dim + 1)

        if safe_shape(res_test, 1) == 1:
            xs = np.diff(y_plot[n_prev + low:n_prev + high])
            ys = np.diff(res_plot[:])
        else:
            xs = np.diff(y_plot[n_prev + low : n_prev + high, dim])
            ys = np.diff(res_plot[low:high, dim])


        plt.plot(xs, ys, 'ko')
        plt.xlabel('actual')
        plt.ylabel('predicted')



plt.show()


# df = get_data(['AAPL', 'QQQ','VZ','NKE','KMI'])
# train_data = df.iloc[0:int(len(df) / 1.5), :]
# test_data = df.iloc[int(len(df) / 1.5):len(df), :]
# dim_reducer = PCA(n_components=100)
# regressor = linear_model.LinearRegression()
# model = Pipeline([('PCA',dim_reducer),('regressor', regressor)])
# test_train_plot(model, train_data, test_data, n_prev=20)
