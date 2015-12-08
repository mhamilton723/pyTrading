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

        if dataY is not None:
            #assert (type(dataX) is type(dataY)) TODO find way to still perform this check
            assert (len(dataX) == len(dataY))

        dlistX, dlistY = [], []
        for i in range(len(dataX) - n_prev):
            if is_pandas:
                dlistX.append(dataX.iloc[i:i + n_prev].as_matrix())
                if dataY is not None:
                    dlistY.append(dataY.iloc[i + n_prev].as_matrix())
                else:
                    dlistY.append(dataX.iloc[i + n_prev].as_matrix())
            else:
                dlistX.append(dataX[i:i + n_prev])
                if dataY is not None:
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
        X_new = self._preprocess(X, Y=None)[0]
        return self.base_estimator.predict(X_new)


def time_series_split(X, test_size=.2, output_numpy=True):
    is_pandas = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
    ntrn = int(len(X) * (1 - test_size))

    if is_pandas:
        X_train = X.iloc[0:ntrn]
        X_test = X.iloc[ntrn:]
    else:
        X_train = X[0:ntrn]
        X_test = X[ntrn:]

    if output_numpy and is_pandas:
        return X_train.as_matrix(), X_test.as_matrix()
    else:
        return X_train, X_test

def safe_shape(array,i):
    try:
        return array.shape[i]
    except IndexError:
        if i > 0:
            return 1
        else:
            raise IndexError

def mse(X1, X2):
    return np.mean((X1 - X2)**2, axis=0)**.5


X = datasets('sp500')
y = X
#X = datasets('synthetic')
#y = X

X_train, X_test = time_series_split(X)
y_train, y_test = time_series_split(y)


windows = range(1, 10)
mses_train = np.empty((len(windows), safe_shape(y_train, 1)))
mses_test = np.empty((len(windows), safe_shape(y_train, 1)))
for i, window in enumerate(windows):
    model = linear_model.LinearRegression()
    time_model = TimeSeriesRegressor(model, window=window)
    time_model.fit(X_train, y_train)
    res_test = time_model.predict(X_test)
    res_train = time_model.predict(X_train)
    mses_train[i, :] = mse(res_train, y_train[window:])
    mses_test[i, :] = mse(res_test, y_test[window:])


plt.boxplot(np.transpose(mses_test))
plt.yscale('log')


plot_train = False
if plot_train: res_plot = res_train; y_plot = y_train
else: res_plot = res_test; y_plot = y_test
low = 0
high = 50#len(res_test)-window

plot_res_vs_actual = False
plot_scatter = False
if plot_res_vs_actual:
    for dim in range(min(safe_shape(res_plot, 1), 4)):
        plt.subplot(2, 2, dim + 1)

        if safe_shape(res_test, 1) == 1:
            plt.plot(res_plot[low:high], label='Predicted')
            plt.plot(y_plot[window + low : window + high], label='Actual')
        else:
            plt.plot(res_plot[low:high, dim], label='Predicted')
            plt.plot(y_plot[window + low : window + high, dim], label='Actual')
        plt.legend(loc='lower left')

if plot_scatter:
    for dim in range(min(safe_shape(res_plot, 1), 4)):
        plt.subplot(2, 2, dim + 1)

        if safe_shape(res_test, 1) == 1:
            xs = np.diff(y_plot[window + low:window + high])
            ys = np.diff(res_plot[:])
        else:
            xs = np.diff(y_plot[window + low : window + high, dim])
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
# test_train_plot(model, train_data, test_data, window=20)
