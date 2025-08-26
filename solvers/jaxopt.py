from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    import jax
    import jaxopt
    import jax.numpy as jnp
    import optax


def loss(beta, data, lmbd):
    X, y = data
    y_X_beta = y * X.dot(beta.flatten())
    l2 = 0.5 * jnp.dot(beta, beta)
    return jnp.log1p(jnp.exp(-y_X_beta)).sum() + lmbd * l2


def _run_lbfgs_solver(X, y, lmbd, n_iter):
    solver = jaxopt.LBFGS(fun=loss, maxiter=n_iter, tol=1e-15)
    beta_init = jnp.zeros(X.shape[1])
    res = solver.run(beta_init, data=(X, y), lmbd=lmbd)
    return res.params


def _run_ncg_solver(X, y, lmbd, n_iter):
    solver = jaxopt.NonlinearCG(fun=loss, maxiter=n_iter, tol=1e-15)
    beta_init = jnp.zeros(X.shape[1])
    res = solver.run(beta_init, data=(X, y), lmbd=lmbd)
    return res.params


def _run_adam_solver(X, y, lmbd, n_iter):
    opt = optax.adam(1e-3)
    solver = jaxopt.OptaxSolver(opt=opt, fun=loss, maxiter=n_iter, tol=1e-15,
                                jit=True, unroll=False)
    beta_init = jnp.zeros(X.shape[1])
    res = solver.run(beta_init, data=(X, y), lmbd=lmbd)
    return res.params


def _run_scipy_solver(X, y, lmbd, n_iter):
    solver = jaxopt.ScipyMinimize(fun=loss, maxiter=n_iter, tol=1e-15,
                                  method='L-BFGS-B')
    beta_init = jnp.zeros(X.shape[1])
    res = solver.run(beta_init, data=(X, y), lmbd=lmbd)
    return res.params


class Solver(BaseSolver):
    name = 'jaxopt'

    install_cmd = 'conda'
    requirements = ['pip:jaxopt', 'pip:optax']

    parameters = {
        'solver': [
            'lbfgs',
            'scipy-lbfgs',
            'ncg',
            'adam',
        ],
    }
    parameter_template = "{solver}"

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = jnp.array(X), jnp.array(y), lmbd
        self.run(1)  # compile jax function

    def run(self, n_iter):
        if self.solver == 'lbfgs':
            _run = _run_lbfgs_solver
        elif self.solver == 'scipy-lbfgs':
            _run = _run_scipy_solver
        elif self.solver == 'ncg':
            _run = _run_ncg_solver
        elif self.solver == 'adam':
            _run = _run_adam_solver
        else:
            raise ValueError(f"Unknown solver {self.solver}")

        if self.solver != 'scipy-lbfgs':
            _run = jax.jit(_run)

        self.coef_ = _run(self.X, self.y, self.lmbd, n_iter)

    def get_result(self):
        return np.array(self.coef_)
