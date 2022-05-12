from benchopt import BaseObjective, safe_import_context

with safe_import_context() as ctx:
    import numpy as np


def _compute_loss(X, y, lmbd, beta, fit_intercept):
    c = 0
    if fit_intercept:
        beta, c = beta[:-1], beta[-1]
    y_X_beta = y * (X.dot(beta.flatten()) + c)
    l2 = 0.5 * np.dot(beta, beta)
    return np.log1p(np.exp(-y_X_beta)).sum() + lmbd * l2


class Objective(BaseObjective):
    name = "L2 Logistic Regression"

    parameters = {
        'lmbd': [0.1, 0.001],
        'fit_intercept': [True, False]
    }

    def set_data(self, X, y, X_test=None, y_test=None):
        self.X, self.y = X, y
        self.X_test, self.y_test = X_test, y_test
        msg = "Logistic loss is implemented with y in [-1, 1]"
        assert set(self.y) == {-1, 1}, msg

    def get_one_solution(self):
        return np.zeros(self.X.shape[1] + self.fit_intercept)

    def compute(self, beta):
        train_loss = _compute_loss(
            self.X, self.y, self.lmbd, beta, self.fit_intercept
        )
        test_loss = None
        if self.X_test is not None:
            test_loss = _compute_loss(
                self.X_test, self.y_test, self.lmbd, beta, self.fit_intercept
            )
        return {"value": train_loss, "Test loss": test_loss}

    def to_dict(self):
        return dict(
            X=self.X, y=self.y, lmbd=self.lmbd,
            fit_intercept=self.fit_intercept
        )
