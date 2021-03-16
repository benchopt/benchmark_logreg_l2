from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.datasets import fetch_openml


class Dataset(BaseDataset):
    name = "madelon"

    install_cmd = 'conda'
    requirements = ['pip:scikit-learn']

    def get_data(self):
        self.X, self.y = fetch_openml("madelon", return_X_y=True)
        # X, y are pandas DataFrames and y is stored as "1", "2"
        self.X, self.y = np.array(self.X), np.array(self.y.astype(int)) - 1
        data = dict(X=self.X, y=self.y)

        return self.X.shape[1], data
