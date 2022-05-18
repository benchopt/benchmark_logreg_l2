from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm
    from sklearn.preprocessing import StandardScaler


class Dataset(BaseDataset):

    name = "criteo"

    install_cmd = 'conda'
    requirements = ['libsvmdata', 'scikit-learn']

    parameters = {
        'scaled': [True, False]
    }

    def get_data(self):
        X, y = fetch_libsvm('criteo')
        X_test, y_test = fetch_libsvm('criteo-test')

        if self.scaled:
            # column scaling - sparse dataset so no mean
            scaler = StandardScaler(with_mean=False)
            X = scaler.fit_transform(X)
            X_test = scaler.transform(X_test)

        y[y == y.min()] = -1
        y[y == y.max()] = 1
        y_test[y_test == y_test.min()] = -1
        y_test[y_test == y_test.max()] = 1

        return dict(X=X, y=y, X_test=X_test, y_test=y_test)
