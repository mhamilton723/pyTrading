
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.pipeline import Pipeline


class TimeSeriesEstimator(BaseEstimator):

    # TODO make this more elegant
    def __init__(self, base_estimator, window=3):
        self.base_estimator = base_estimator
        self.window = window

    def __repr__(self):
        return "TimeSeriesEstimator: "+repr(self.base_algorithm)

    def _window_dataset(self,data, n_prev=1):
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

    def _window_dataset_sklearn(self,df, window=3, pandas=True):
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



class TimeSeriesRegressor(TimeSeriesEstimator):
    # __slots__ = "fit_algorithm"

    # def __init__(self, **kwargs):
    #    super(MetaRegressor, self).__init__(**kwargs)

    def fit(self, X, Y=None):
        # self._update_algorithm()
        return self.base_estimator.fit(X, Y)

    def score(self, X, Y, **kwargs):
        return self.base_estimator.score(X, Y, **kwargs)

    def predict(self, X):
        return self.base_estimator.predict(X)
