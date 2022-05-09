from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    # Dependencies of download_libsvm are scikit-learn, download and tqdm
    from libsvmdata import fetch_libsvm


class Dataset(BaseDataset):
    name = "rcv1"
    is_sparse = True

    install_cmd = 'conda'
    requirements = ['pip:libsvmdata']

    def __init__(self):
        self.X, self.y = None, None

    def get_data(self):

        if self.X is None:
            self.X, self.y = fetch_libsvm('rcv1.binary', min_nnz=0)
            self.X_test, self.y_test = fetch_libsvm(
                'rcv1.binary_test', min_nnz=0
            )

        return dict(X=self.X, y=self.y, X_test=self.X_test, y_test=self.y_test)
