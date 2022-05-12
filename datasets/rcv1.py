from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    # Dependencies of download_libsvm are scikit-learn, download and tqdm
    from libsvmdata import fetch_libsvm


class Dataset(BaseDataset):
    name = "rcv1"
    is_sparse = True

    install_cmd = 'conda'
    requirements = ['pip:libsvmdata']

    parameters = {
        'scaled': [True, False]
    }

    def get_data(self):

        X, y = fetch_libsvm('rcv1.binary', min_nnz=0)
        X_test, y_test = fetch_libsvm('rcv1.binary_test', min_nnz=0)

        if self.scaled:
            # column scaling
            mu, sigma = X.mean(axis=0), X.std(axis=0)
            X -= mu
            X /= sigma
            X_test -= mu
            X_test /= sigma

        data = dict(X=X, y=y, X_test=X_test, y_test=y_test)

        return data
