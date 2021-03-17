from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.datasets import fetch_openml


class Dataset(BaseDataset):
    name = "madelon"

    install_cmd = 'conda'
    requirements = ['pip:scikit-learn']

    def get_data(self):
        X, y = fetch_openml("madelon", return_X_y=True, as_frame=False)
        # y is stored as "1", "2" and needs to be moved to [-1, 1]
        y = y.astype(int)
        y[y == 1] = -1
        y[y == 2] = 1
        data = dict(X=X, y=y)

        return X.shape[1], data
