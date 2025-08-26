from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm


class Dataset(BaseDataset):
    name = "madelon"

    install_cmd = 'conda'
    requirements = ['pip:libsvmdata']

    def get_data(self):
        X, y = fetch_libsvm("madelon")
        X_test, y_test = fetch_libsvm("madelon_test")
        return dict(X=X, y=y, X_test=X_test, y_test=y_test)
