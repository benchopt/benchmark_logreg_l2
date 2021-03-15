from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from copt.datasets import load_covtype


class Dataset(BaseDataset):
    name = "covtype"

    install_cmd = 'conda'
    requirements = ['pip:copt']

    def __init__(self):
        self.X, self.y = None, None

    def get_data(self):

        if self.X is None:
            self.X, self.y = load_covtype()
            self.X = np.array(self.X.todense())

        data = dict(X=self.X, y=self.y)

        return self.X.shape[1], data
