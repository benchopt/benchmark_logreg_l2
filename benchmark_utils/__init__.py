import json
import warnings

import numpy as np
from scipy import sparse
from benchopt.config import get_data_path
from sklearn.linear_model import LogisticRegressionCV


def get_lmbd_cv(dataset, X, y):
    """Return the lmbd selected by cross-validation for each fit_intercept.

    The values are computed once, when preparing the dataset, and stored in
    <data_home>/lmbd_cv/<dataset name>.json, keyed by the dataset parameters
    and then by str(fit_intercept).
    """
    path = get_data_path("lmbd_cv") / f"{dataset.name}.json"
    key = str(dataset)
    if path.exists() and key in (lmbd_cv := json.loads(path.read_text())):
        return lmbd_cv[key]

    print(f"\nSelecting lmbd_cv for {key} by cross-validation, stored in "
          f"{path}", flush=True)
    # The grid is relative to L / 4, the smoothness constant of the loss, so
    # that it follows the scale of the features.
    if sparse.issparse(X):
        L = sparse.linalg.svds(X, k=1)[1][0] ** 2
    else:
        L = np.linalg.norm(X, ord=2) ** 2
    Cs = 4 / L * np.logspace(-2, 16, 37)

    lmbd = {}
    for fit_intercept in [False, True]:
        warnings.filterwarnings("ignore", category=FutureWarning)
        clf = LogisticRegressionCV(
            Cs=Cs, solver="newton-cg",
            scoring="neg_log_loss", fit_intercept=fit_intercept, n_jobs=-1
        ).fit(X, y)
        # sklearn minimizes C * sum(loss) + ||w||^2 / 2, so lmbd = 1 / C.
        lmbd[str(fit_intercept)] = 1 / clf.C_[0]

    # Re-read the file to keep the values written meanwhile by other runs.
    lmbd_cv = json.loads(path.read_text()) if path.exists() else {}
    lmbd_cv[key] = lmbd
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lmbd_cv, indent=2, sort_keys=True))
    return lmbd
