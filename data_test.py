import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from TimeSeriesEstimator import TimeSeriesRegressor, time_series_split, cascade_cv
from utils import load_s_and_p_data, cache

__author__ = 'Mark'
X = load_s_and_p_data(start="2009-1-1", only_close=True)
names = list(X.columns.values)

@cache('data/data_dependence_cache.pkl')
def get_data_dependece(X, data_sizes, folds=20, test_size=30, n_prev=2, log=True):
    bests = np.empty((len(data_sizes), folds, X.shape[1]))
    for i, data_size in enumerate(data_sizes):
        pairs = cascade_cv(len(X), folds, data_size=data_size, test_size=test_size, number=True)
        for j, pair in enumerate(pairs):
            if log:
                print('data size: {} trial {} '.format(data_size, j))
            X_train, X_test = np.array(X.iloc[pair[0], :]), np.array(X.iloc[pair[1], :])
            tsr = TimeSeriesRegressor(LinearRegression(), n_prev=n_prev)
            tsr.fit(X_train)
            fc = tsr.forecast(X_train, len(X_test))

            def changes(X, start=0, end=-1):
                return np.array([X[end, i] - X[start, i] for i in range(X.shape[1])])

            best_is = changes(fc).argsort()[::-1]
            for k in range(X.shape[1]):
                bests[i, j, k] = changes(X_test)[best_is[k]] - np.mean(changes(X_test))

    return bests


data_sizes = [.1, .2, .3, .4]
bests = get_data_dependece(X, data_sizes)

x = [int(ds*X.shape[0]) for ds in data_sizes]
LR = np.transpose(bests[:, :, 0])
SLR = np.transpose(bests[:, :, 1])
plt.plot(x, np.mean(LR, axis=0), color='b', label='Largest Rise')
plt.plot(x, np.mean(SLR, axis=0), color='g', label='Second Largest Rise')

plt.title('Average performance of Linear TSE over 20 Different Datasets')
plt.xlabel('Size of dataset')
plt.ylabel('mean(Price Change - mean(Price Change))')
plt.legend()
plt.show()
