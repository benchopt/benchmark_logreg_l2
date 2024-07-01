import sys  # noqa: F401
import pytest  # noqa: F401

from benchopt.utils.sys_info import get_cuda_version


def is_numpy_2():
    import numpy as np
    return np.__version__ > '2'


def check_test_solver_install(solver_class):
    """Hook called in `test_solver_install`.

    If one solver needs to be skip/xfailed on some
    particular architecture, call pytest.xfail when
    detecting the situation.
    """
    cuda_version = get_cuda_version()
    is_platform_macOS = sys.platform == "darwin"

    if solver_class.name.lower() == "cuml":
        if is_platform_macOS:
            pytest.xfail("Cuml is not supported on MacOS.")
        if cuda_version is None:
            pytest.xfail("Cuml needs a working GPU hardware.")

    if solver_class.name.lower() == 'snapml':
        if is_platform_macOS:
            pytest.skip(
                "Running snapml on MacOS takes a lot of time.\n"
                "See PR 38 in benchopt/benchmark_logreg_l2"
            )
        if is_numpy_2():
            pytest.skip(
                "SnapML is not supported with numpy >= 2.\n"
                "See benchopt/benchmark_logreg_l2#51 for tracking."
            )

    if solver_class.name.lower() == 'copt':
        if is_numpy_2():
            pytest.skip(
                "SnapML is not supported with numpy >= 2.\n"
                "See benchopt/benchmark_logreg_l2#51 for tracking."
            )
