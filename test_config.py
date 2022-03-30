import sys  # noqa: F401

import pytest  # noqa: F401


def check_test_solver_install(solver_class):
    """Hook called in `test_solver_install`.

    If one solver needs to be skip/xfailed on some
    particular architecture, call pytest.xfail when
    detecting the situation.
    """
    if solver_class.name.lower() == "cuml" and sys.platform == "darwin":
        pytest.xfail("Cuml is not supported on MacOS.")
