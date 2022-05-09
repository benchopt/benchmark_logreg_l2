from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    # Dependencies of download_libsvm are scikit-learn, download and tqdm
    from libsvmdata import fetch_libsvm


class Dataset(BaseDataset):
    name = "ijcnn1"
    is_sparse = True

    install_cmd = "conda"
    requirements = ["pip:libsvmdata"]

    def __init__(self):
        self.X, self.y = None, None

    def get_data(self):

        if self.X is None:
            self.X, self.y = fetch_libsvm("ijcnn1")

        return dict(X=self.X, y=self.y)
