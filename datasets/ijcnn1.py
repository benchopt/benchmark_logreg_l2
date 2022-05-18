from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    # Dependencies of download_libsvm are scikit-learn, download and tqdm
    from libsvmdata import fetch_libsvm
    from sklearn.preprocessing import StandardScaler


class Dataset(BaseDataset):
    name = "ijcnn1"

    install_cmd = "conda"
    requirements = ["pip:libsvmdata"]

    parameters = {
        'scaled': [True, False]
    }

    def get_data(self):
        X, y = fetch_libsvm("ijcnn1")
        X_test, y_test = fetch_libsvm('ijcnn1_test')

        if self.scaled:
            # column scaling
            scaler = StandardScaler(with_mean=False)
            X = scaler.fit_transform(X)
            X_test = scaler.transform(X_test)

        return dict(X=X, y=y, X_test=X_test, y_test=y_test)
