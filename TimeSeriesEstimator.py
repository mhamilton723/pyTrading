import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from utils import datasets


class TimeSeriesEstimator(BaseEstimator):
    def __init__(self, base_estimator, window=3):
        self.base_estimator = base_estimator
        self.window = window

    def __repr__(self):
        return "TimeSeriesEstimator: " + repr(self.base_algorithm)

    def _window_dataset(self, n_prev, dataX, dataY=None):
        """
        converts a dataset into an autocorrelation dataset with number of previous time steps = n_prev
        returns a an X dataset of shape (samples,timesteps,features) and a Y dataset of shape (samples,features)
        """
        is_pandas = isinstance(dataX, pd.DataFrame)

        if dataY:
            assert (type(dataX) is type(dataY))
            assert (len(dataX) == len(dataY))

        dlistX, dlistY = [], []
        for i in range(len(dataX) - n_prev):
            if is_pandas:
                dlistX.append(dataX.iloc[i:i + n_prev].as_matrix())
                if dataY:
                    dlistY.append(dataY.iloc[i + n_prev].as_matrix())
                else:
                    dlistY.append(dataX.iloc[i + n_prev].as_matrix())
            else:
                dlistX.append(dataX[i:i + n_prev])
                if dataY:
                    dlistY.append(dataY[i + n_prev])
                else:
                    dlistY.append(dataX[i + n_prev])

        darrX = np.array(dlistX)
        darrY = np.array(dlistY)
        return darrX, darrY

    def _unravel_window_data(self, data):
        dlist = []
        for i in range(data.shape[0]):
            dlist.append(data[i, :, :].ravel())
        return np.array(dlist)

    def _preprocess(self, X, Y):
        X_wind, Y_data = self._window_dataset(self.window, X, Y)
        X_data = self._unravel_window_data(X_wind)
        return X_data, Y_data

    def fit(self, X, Y=None):
        ''' X and Y are datasets in chronological order, or X is a time series '''

        return self.base_estimator.fit(*self._preprocess(X, Y))


class TimeSeriesRegressor(TimeSeriesEstimator):
    def score(self, X, Y, **kwargs):
        return self.base_estimator.score(*self._preprocess(X, Y), **kwargs)

    def predict(self, X):
        return self.base_estimator.predict(self._preprocess(X, Y=None)[0])


def time_series_split(df, test_size=.2, output_numpy=True):
    is_pandas = isinstance(df, pd.DataFrame)
    ntrn = int(len(df) * (1 - test_size))

    if is_pandas:
        X_train = df.iloc[0:ntrn]
        X_test = df.iloc[ntrn:]
    else:
        X_train = df[0:ntrn]
        X_test = df[ntrn:]

    if output_numpy and is_pandas:
        return X_train.as_matrix(), X_test.as_matrix()
    else:
        return X_train, X_test


window = 1
df = datasets('sp500')
model = linear_model.LinearRegression()
time_model = TimeSeriesRegressor(model, window=window)
X_train, X_test = time_series_split(df)
time_model.fit(X_train)
res_test = time_model.predict(X_test)
res_train = time_model.predict(X_train)

plot_train = False
for dim in range(min(res_test.shape[1], 4)):
    plt.subplot(2, 2, dim + 1)
    if plot_train:
        plt.plot(res_train[:, dim], label='Results test')
        plt.plot(X_train[window:, dim], label='Test')
    else:
        plt.plot(res_test[:, dim], label='Results test')
        plt.plot(X_test[window:, dim], label='Test')
    plt.legend(loc='upper left')


plt.show()


# df = get_data(['AAPL', 'QQQ','VZ','NKE','KMI'])
# train_data = df.iloc[0:int(len(df) / 1.5), :]
# test_data = df.iloc[int(len(df) / 1.5):len(df), :]
# dim_reducer = PCA(n_components=100)
# regressor = linear_model.LinearRegression()
# model = Pipeline([('PCA',dim_reducer),('regressor', regressor)])
# test_train_plot(model, train_data, test_data, window=20)
