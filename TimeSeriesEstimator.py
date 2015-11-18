
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.pipeline import Pipeline


class MetaEstimator(BaseEstimator):
    # __slots__ = "algorithm","parameters"

    # TODO make this more elegant
    def __init__(self, **kwargs):
        if 'base_algorithm' in kwargs:
            algo = kwargs.pop('base_algorithm')
            self.base_algorithm = algo
        else:
            self.base_algorithm = None

        self.parameters = {}
        if kwargs:
            self.parameters = utils.dict_merge(self.parameters, kwargs)

    def __repr__(self):
        return "MetaEstimator: "+repr(self.base_algorithm)

    def set_params(self, **params):
        if 'base_algorithm' in params:
            algo = params.pop('base_algorithm')
            self.base_algorithm = algo
        if not self.parameters:
            self.parameters = {}
        if params:
            self.parameters = self.parameters = utils.dict_merge(self.parameters, params)
        if self.base_algorithm and self.parameters:
            self._update_algorithm()
        return self

    def _update_algorithm(self):
        """Make and configure a copy of the `base_estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        if self.base_algorithm and self.parameters:
            self.base_algorithm = self.base_algorithm.set_params(**self.parameters)
        elif not self.base_algorithm:
            raise ValueError("No algorithm found")
        elif not self.parameters:
            print("WARNING: No parameters found")

    def get_params(self, deep=True):
        out = {}
        if self.base_algorithm is not None:
            dict1 = self.base_algorithm.get_params(deep)
            out.update(dict1)
            out['base_algorithm'] = self.base_algorithm
        if self.parameters:
            dict2 = self.parameters
            out.update(dict2)
        return out


class MetaRegressor(MetaEstimator):
    # __slots__ = "fit_algorithm"

    # def __init__(self, **kwargs):
    #    super(MetaRegressor, self).__init__(**kwargs)

    def fit(self, X, Y=None):
        # self._update_algorithm()
        return self.base_algorithm.fit(X, Y)

    def score(self, X, Y, **kwargs):
        return self.base_algorithm.score(X, Y, **kwargs)

    def predict(self, X):
        return self.base_algorithm.predict(X)
