from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from tick.linear_model import ModelLogReg
    from tick.prox import ProxL2Sq
    from tick.solver import SVRG


class Solver(BaseSolver):
    name = 'svrg-tick'

    install_cmd = 'conda'
    requirements = ['pip:tick']
    references = [
        'Bacry, Emmanuel, et al. "Tick: a Python library for statistical '
        'learning, with a particular emphasis on time-dependent modelling." '
        'arXiv preprint arXiv:1707.03003 (2017).'
    ]

    # Tick can take sparse matrix as input but this does not converge...
    support_sparse = False

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.fit_intercept = fit_intercept
        self.model = ModelLogReg(fit_intercept=fit_intercept).fit(X, y)
        # Scaled with 1/n_samples as logistic objective is taken with mean
        prox = ProxL2Sq(lmbd / X.shape[0])
        self.clf = SVRG(
            tol=1e-16, verbose=False, step_type='fixed', record_every=int(1e8),
            rand_type='perm'
        )
        self.clf.set_model(self.model).set_prox(prox)

    def run(self, n_iter):
        optimal_step_size = 1 / self.model.get_lip_max()
        self.clf.max_iter = n_iter
        self.coef_ = self.clf.solve(step=optimal_step_size)

    def get_result(self):
        coef = self.coef_.ravel()
        if self.fit_intercept:
            coef = np.r_[coef, self.clf.intercept_]
        return coef

    def get_next(self, stop_val):
        return stop_val + 10
