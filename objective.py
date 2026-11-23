from benchopt import BaseObjective

import numpy as np


def _compute_loss(X, y, lmbd, beta, fit_intercept):
    c = 0
    if fit_intercept:
        beta, c = beta[:-1], beta[-1]
    y_X_beta = y * (X.dot(beta.flatten()) + c)
    l2 = 0.5 * np.dot(beta, beta)

    # This should be more stable than log1p as we don't need to take the exp
    loss = np.logaddexp(0, -y_X_beta).sum() + lmbd * l2
    error = (np.sign(y_X_beta) <= 0).mean()
    return loss, error


class Objective(BaseObjective):
    name = "L2 Logistic Regression"
    min_benchopt_version = "1.10.1"
    # LogisticRegressionCV(use_legacy_attributes=...) to select lmbd_cv.
    requirements = ["pip::scikit-learn>=1.8"]

    parameters = {
        'lmbd': [0.1, 1.0, 10.0],
        'fit_intercept': [False, True]
    }

    def set_data(self, X, y, lmbd_cv, X_test=None, y_test=None):
        self.X, self.y = X, y
        self.X_test, self.y_test = X_test, y_test
        # lmbd is relative to the value selected by cross-validation.
        self.lmbd_cv = lmbd_cv[str(self.fit_intercept)]
        msg = "Logistic loss is implemented with y in [-1, 1]"
        assert set(self.y) == {-1, 1}, msg

    def get_one_result(self):
        return dict(beta=np.zeros(self.X.shape[1] + self.fit_intercept))

    def evaluate_result(self, beta):
        lmbd = self.lmbd * self.lmbd_cv
        train_loss, train_error = _compute_loss(
            self.X, self.y, lmbd, beta, self.fit_intercept
        )
        test_loss = test_error = None
        if self.X_test is not None:
            test_loss, test_error = _compute_loss(
                self.X_test, self.y_test, lmbd, beta, self.fit_intercept
            )
        return {
            "Train loss": train_loss,
            "Train error": train_error,
            "Test loss": test_loss,
            "Test error": test_error,
            "value": train_loss
        }

    def get_objective(self):
        return dict(
            X=self.X, y=self.y, lmbd=self.lmbd * self.lmbd_cv,
            fit_intercept=self.fit_intercept
        )
