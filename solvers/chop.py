import warnings
import os
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    import chop


class Solver(BaseSolver):
    name = 'chop'

    install_cmd = 'conda'
    requirements = ['pip:https://github.com/openopt/chop/archive/master.zip']

    parameters = {'solver': ['pgd'],
                  'line_search': [False, True],
                  'device': ['cuda', 'cpu']}

    def skip(self, X, y, lmbd):
        if self.device == 'cuda' and not torch.cuda.is_available():
            return True, "CUDA is not available."
        return False, None

    def set_objective(self, X, y, lmbd):
        self.lmbd = lmbd

        if self.device == 'cpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

        self.X = torch.tensor(X)
        self.y = torch.tensor(y > 0, dtype=torch.float64)

    def run(self, n_iter):
        X, y, solver = self.X, self.y, self.solver
        # Decide whether to solve a batch of problems
        n_points, n_features = X.shape

        x0 = torch.zeros(1, n_features, requires_grad=True, dtype=X.dtype)
        if n_iter == 0:
            self.beta = x0
            return

        criterion = torch.nn.BCEWithLogitsLoss()

        @chop.utils.closure
        def logloss(x):
            alpha = self.lmbd / X.size(0)
            out = chop.utils.bmv(X, x)
            loss = criterion(out, y)
            reg = .5 * alpha * chop.utils.bdot(x, x).squeeze()
            return loss + reg

        warnings.filterwarnings('ignore', category=RuntimeWarning)

        if solver == 'pgd':
            if self.line_search:
                step = 'backtracking'
            else:
                # estimate the step using backtracking line search once
                step = None

            result = chop.optim.minimize_pgd(logloss, x0,
                                             prox=lambda x, s: x,
                                             step=step,
                                             max_iter=n_iter)
        else:
            raise NotImplementedError

        self.beta = result.x

    def get_result(self):
        return self.beta.cpu().detach().numpy().flatten()
