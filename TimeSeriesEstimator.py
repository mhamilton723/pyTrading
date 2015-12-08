import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class TimeSeriesEstimator(BaseEstimator):
    def __init__(self, base_estimator, n_prev=3, n_ahead=1):
        self.base_estimator = base_estimator
        self.n_prev = n_prev
        self.n_ahead = n_ahead

    def __repr__(self):
        return "TimeSeriesEstimator: " + repr(self.base_algorithm)

    def _window_dataset(self, n_prev, dataX, dataY=None, n_ahead=1):
        """
        converts a dataset into an autocorrelation dataset with number of previous time steps = n_prev
        returns a an X dataset of shape (samples,timesteps,features) and a Y dataset of shape (samples,features)
        """
        is_pandas = isinstance(dataX, pd.DataFrame)

        if dataY is not None:
            #assert (type(dataX) is type(dataY)) TODO find way to still perform this check
            assert (len(dataX) == len(dataY))

        dlistX, dlistY = [], []
        for i in range(len(dataX) - n_prev + 1 - n_ahead):
            if is_pandas:
                dlistX.append(dataX.iloc[i:i + n_prev].as_matrix())
                if dataY is not None:
                    dlistY.append(dataY.iloc[i + n_prev - 1 + n_ahead].as_matrix())
                else:
                    dlistY.append(dataX.iloc[i + n_prev - 1 + n_ahead].as_matrix())
            else:
                dlistX.append(dataX[i:i + n_prev])
                if dataY is not None:
                    dlistY.append(dataY[i + n_prev - 1 + n_ahead])
                else:
                    dlistY.append(dataX[i + n_prev - 1 + n_ahead])

        darrX = np.array(dlistX)
        darrY = np.array(dlistY)
        return darrX, darrY

    def _unravel_window_data(self, data):
        dlist = []
        for i in range(data.shape[0]):
            dlist.append(data[i, :, :].ravel())
        return np.array(dlist)

    def _preprocess(self, X, Y):
        X_wind, Y_data = self._window_dataset(self.n_prev, X, Y, self.n_ahead)
        X_data = self._unravel_window_data(X_wind)
        return X_data, Y_data

    def fit(self, X, Y=None):
        ''' X and Y are datasets in chronological order, or X is a time series '''

        return self.base_estimator.fit(*self._preprocess(X, Y))


class TimeSeriesRegressor(TimeSeriesEstimator,RegressorMixin):
    def score(self, X, Y, **kwargs):
        return self.base_estimator.score(*self._preprocess(X, Y), **kwargs)

    def predict(self, X):
        X_new = self._preprocess(X, Y=None)[0]
        return self.base_estimator.predict(X_new)

