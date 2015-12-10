import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from utils import datasets, safe_shape, mse
from TimeSeriesEstimator import TimeSeriesRegressor, time_series_split, time_series_cv, cascade_cv
from sklearn.cross_validation import KFold


def train_test_plot(pred_train, y_train, pred_test, y_test, n_prev, titles, cap=4):
    output_dim = 1 if len(y_test.shape) == 1 else y_test.shape[1]
    if output_dim > cap:
        output_dim = cap
        print(output_dim)

    for i in range(output_dim):
        plt.subplot(output_dim, 2, 2 * i + 1)
        if output_dim == 1:
            plt.plot(pred_train, 'r', label="Predicted")
            plt.plot(y_train[n_prev:], 'b--', label="Actual")
        else:
            plt.plot(pred_train[:, i], 'r', label="Predicted")
            plt.plot(y_train[n_prev:, i], 'b--', label="Actual")
        # nprev: because the first predicted point needed n_prev steps of data
        # plt.title("Training performance of " + titles[i])
        # plt.legend(loc='lower right')

        plt.subplot(output_dim, 2, 2 * i + 2)
        if output_dim == 1:
            plt.plot(pred_test, 'r', label="Predicted")
            plt.plot(y_test[n_prev:], 'b--', label="Actual")
        else:
            plt.plot(pred_test[:, i], 'r', label="Predicted")
            plt.plot(y_test[n_prev:, i], 'b--', label="Actual")
            # nprev: because the first predicted point needed n_prev steps of data
            # plt.title("Testing performance of " + titles[i])
            # plt.legend(loc='lower left')

    plt.gcf().set_size_inches(15, 6)
    plt.show()


X = datasets('sp500')
y = X['AAPL']
X_train, X_test = time_series_split(X)
y_train, y_test = time_series_split(y)

n_prev = 3
tsr = TimeSeriesRegressor(Lasso(), n_prev=n_prev)


param_grid = [{'alpha': [.01, .05, .1, .5, 1, 5, 10]}]
cv = cascade_cv(len(X_train), n_folds=5)
grid = GridSearchCV(tsr, param_grid, cv=cv)
grid.fit(X_train, y_train)
pred_train_3 = grid.predict(X_train)  # outputs a numpy array of length: len(X_train)-n_prev
pred_test_3 = grid.predict(X_test)
print(grid.grid_scores_)

print(grid.best_params_)

train_test_plot(pred_train_3, y_train, pred_test_3, y_test, n_prev, 'Optimized Lasso TSR on APPLE')

# train_test_plot(pred_train, y_train, pred_test, y_test, n_prev, ['A','AA','AAL','AAP'])
