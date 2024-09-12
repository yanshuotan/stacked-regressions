import numpy as np
import cvxpy as cp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class NestedDataProblem:

    def __init__(self, X, y, nested_sets) -> None:
        self.X = X
        self.y = y
        self.n = len(y)
        self.n_regressions = len(nested_sets)
        for k in range(self.n_levels - 1):
            assert nested_sets[k].is_subset(nested_sets[k + 1])
        self.nested_sets = nested_sets
        self.d = np.array([len(index_set) for index_set in nested_sets])
        self.regressions = [None] * self.n_regressions
        self.errors = [None] * self.n_regressions

    def get_X_subset(self, k):
        return self.X[self.nested_sets[k]]

    def fit_regressions(self):
        for k in range(self.n_regressions):
            lr = LinearRegression()
            X_sub = self.get_X_subset(k)
            lr.fit(X_sub, self.y)
            self.regressions[k] = lr
            self.predictions[k] = lr.predict(X_sub)
            self.errors = mean_squared_error(y, lr.predict(X_sub))
            
    def fit_model_selection(self, l, sigma_sq):
        self.scores = self.errors + l * sigma_sq * self.d / self.n
        self.best_model_idx = np.argmin(self.scores)
        self.best_model = self.regressions[self.best_model_idx]

    def fit_stacking(self, l, tau, sigma_sq):
        alpha = cp.Variable(self.n_regressions)
        objective = cp.Minimize(mean_squared_error(self.y, self._stacking_predict(alpha)) + self._stacking_peanlty(alpha, l, tau, sigma_sq))
        constraints = [0 <= alpha]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        alpha_hat = problem.value()
        
    
    def _stacking_predict(self, alpha):
        return np.sum([alpha[k] * self.predictions[k] for k in range(self.n_regressions)], axis=0)
    
    def _stacking_penalty(self, alpha, l, tau, sigma_sq):
        dof = np.sum([alpha[k] * self.d[k] for k in self.n_regressions])
        dim_alpha = self.d[np.nonzero(alpha)[0][-1]]
        return 2 * sigma_sq * dof / self.n + max(0, l - tau) ** 2 / l * sigma_sq ** 2 / self.n * dim_alpha
            

class StackedRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, base_regressors, meta_regressor):
        self.base_regressors = base_regressors
        self.meta_regressor = meta_regressor

    def fit(self, X, y):
        # Fit the base regressors
        for regressor in self.base_regressors:
            regressor.fit(X, y)

        # Prepare the meta features
        meta_features = np.column_stack([regressor.predict(X) for regressor in self.base_regressors])

        # Fit the meta regressor
        self.meta_regressor.fit(meta_features, y)

    def predict(self, X):
        # Generate meta features
        meta_features = np.column_stack([regressor.predict(X) for regressor in self.base_regressors])

        # Make predictions using the meta regressor
        return self.meta_regressor.predict(meta_features)