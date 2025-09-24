from benchopt import BaseSolver

import numpy as np
from lightning import classification
from lightning.impl.dataset_fast import get_dataset
from lightning.impl.sag_fast import get_auto_step_size


class Solver(BaseSolver):
    name = 'Lightning'

    install_cmd = 'conda'
    requirements = [
        'pip:git+https://github.com/scikit-learn-contrib/lightning.git@master'
    ]

    stopping_strategy = "callback"

    parameters = {
        'method': [
            'cd', 'saga', 'sag'  # , 'svrg'
        ]
    }

    def skip(self, X, y, lmbd, fit_intercept):
        if fit_intercept:
            return True, "no implemented with fit_intercept"

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):

        self.X, self.y, self.lmbd = X, y, lmbd

        if self.method == "cd":
            self.clf = classification.CDClassifier(
                loss='log', penalty='l2', C=1, alpha=self.lmbd,
                tol=0, permute=False, shrinking=False, warm_start=False,
                max_iter=int(1e8))
        elif self.method == "saga":
            self.clf = classification.SAGAClassifier(
                loss='log', alpha=self.lmbd / X.shape[0], tol=0,
                max_iter=int(1e8)
            )
        elif self.method == "sag":
            self.clf = classification.SAGClassifier(
                loss='log', alpha=self.lmbd / X.shape[0], tol=0,
                max_iter=int(1e8)
            )
        elif self.method == "svrg":
            # Not working for now
            n_samples, _ = X.shape
            loss, alpha = 'log', self.lmbd / n_samples
            ds = get_dataset(X, order='c')
            eta = get_auto_step_size(
                ds, alpha, loss, 1, sample_weight=np.ones(n_samples)
            )
            self.clf = classification.SVRGClassifier(
                eta=eta, loss=loss, alpha=alpha, tol=0,
                max_iter=int(1e8)
            )
        else:
            raise ValueError(f"Unknown method {self.method}.")

    def run(self, callback):

        def cb(clf):
            should_stop = not callback(clf.coef_.flatten())
            if should_stop:
                return True

        self.clf.callback = cb
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return dict(beta=self.clf.coef_.flatten())
