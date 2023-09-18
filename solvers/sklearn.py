import warnings


from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import SGDClassifier
    from scipy.optimize.linesearch import LineSearchWarning


class Solver(BaseSolver):
    name = 'sklearn'

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    parameters = {
        'solver': [
            'liblinear',
            'newton-cg',
            'lbfgs',
            'sag',
            'saga',
            'sgd',
        ],
    }
    parameter_template = "{solver}"

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=LineSearchWarning)
        warnings.filterwarnings('ignore', category=UserWarning,
                                message='Line Search failed')

        if self.solver == 'sgd':
            self.clf = SGDClassifier(
                loss="log", alpha=self.lmbd / (X.shape[0] * 2.0),
                penalty='l2', fit_intercept=False, tol=1e-15,
                random_state=42, eta0=.01, learning_rate="constant"
            )
        else:
            self.clf = LogisticRegression(
                solver=self.solver, C=1 / self.lmbd,
                penalty='l2', fit_intercept=False, tol=1e-15
            )

    def run(self, n_iter):
        if self.solver == "sgd":
            n_iter += 1
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return dict(beta=self.clf.coef_.flatten())
