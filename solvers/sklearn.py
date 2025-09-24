import warnings

from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier


class Solver(BaseSolver):
    name = 'sklearn'

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    parameters = {
        'solver': [
            'liblinear',
            'newton-cg',
            'newton-cholesky',
            'lbfgs',
            'sag',
            'saga',
            'sgd',
        ],
    }
    parameter_template = "{solver}"
    stopping_criterion = SufficientProgressCriterion(
        eps=1e-12, patience=5, strategy='iteration'
    )

    def skip(self, X, y, lmbd, fit_intercept):
        if len(np.unique(y)) != 2 and self.solver == 'newton-cholesky':
            return True, "Newton-Cholesky only works for binary classification"
        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning,
                                message='Line Search failed')

        if self.solver == 'sgd':
            self.clf = SGDClassifier(
                loss="log", alpha=self.lmbd / (X.shape[0] * 2.0),
                penalty='l2', fit_intercept=fit_intercept, tol=1e-15,
                random_state=42, eta0=.01, learning_rate="constant"
            )
        else:
            self.clf = LogisticRegression(
                solver=self.solver, C=1 / self.lmbd,
                penalty='l2', fit_intercept=fit_intercept, tol=1e-15
            )

    def run(self, n_iter):
        if self.solver == "sgd":
            n_iter += 1
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        coef = self.clf.coef_.flatten()
        if self.clf.fit_intercept:
            coef = np.r_[coef, self.clf.intercept_]
        return dict(beta=coef)
