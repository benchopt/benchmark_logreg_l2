from benchopt import BaseSolver
from benchopt.helpers.requires_gpu import gpu_available
from benchopt.stopping_criterion import SufficientProgressCriterion

import torch
from scipy import sparse


class Solver(BaseSolver):
    """torch.optim.LBFGS with a strong Wolfe line search, on GPU."""
    name = "torch-lbfgs"

    requirements = ["pip::torch"]

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-12, patience=5, strategy='iteration'
    )

    def skip(self):
        if not gpu_available(self):
            return "needs a GPU"
        return None

    def set_objective(self, X, y, lmbd, fit_intercept):
        if sparse.issparse(X):
            X = X.tocsr()
            self.X = torch.sparse_csr_tensor(
                X.indptr, X.indices, X.data, X.shape, device="cuda"
            )
        else:
            self.X = torch.as_tensor(X, device="cuda")
        # Keep the dtype of the data, and use it for y and the iterates.
        self.y = torch.as_tensor(y, dtype=self.X.dtype, device="cuda")
        self.lmbd, self.fit_intercept = lmbd, fit_intercept
        self.n_features = X.shape[1]

        # Run once per problem to absorb its setup costs (jit, CUDA) from the
        # timings, as warm_up only runs once per solver.
        self.run(1)

    def loss(self, w):
        beta, c = w[:self.n_features], w[self.n_features:].sum()
        y_X_beta = self.y * (self.X @ beta + c)
        return (
            torch.logaddexp(torch.zeros_like(y_X_beta), -y_X_beta).sum()
            + self.lmbd / 2 * beta @ beta
        )

    def run(self, n_iter):
        w = torch.zeros(
            self.n_features + self.fit_intercept, dtype=self.X.dtype,
            device="cuda", requires_grad=True
        )
        if n_iter:
            optimizer = torch.optim.LBFGS(
                [w], max_iter=n_iter, tolerance_grad=0, tolerance_change=0,
                line_search_fn="strong_wolfe"
            )

            def closure():
                optimizer.zero_grad()
                loss = self.loss(w)
                loss.backward()
                return loss

            optimizer.step(closure)
        torch.cuda.synchronize()
        self.coef = w.detach()

    def get_result(self):
        return dict(beta=self.coef.cpu().numpy())
