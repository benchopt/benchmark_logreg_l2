from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.datasets import fetch_covtype


class Dataset(BaseDataset):
    name = "covtype"

    install_cmd = 'conda'
    requirements = ['pip:copt']

    def get_data(self):
        self.X, self.y = fetch_covtype(return_X_y=True)

        data = dict(X=self.X, y=self.y)

        return self.X.shape[1], data
