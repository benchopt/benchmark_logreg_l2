from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.datasets import fetch_covtype
    from sklearn.preprocessing import StandardScaler


class Dataset(BaseDataset):
    name = "covtype_binary"

    install_cmd = 'conda'
    requirements = ['pip:scikit-learn']

    parameters = {
        'standardized': [False, True]
    }

    def get_data(self):
        X, y = fetch_covtype(return_X_y=True)
        y[y != 2] = -1
        y[y == 2] = 1  # try to separate class 2 from the other 6 classes.

        if self.standardized:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        data = dict(X=X, y=y)

        return X.shape[1], data
