import sys  # noqa: F401
import pytest  # noqa: F401

from benchopt.utils.sys_info import get_cuda_version


def check_test_solver_install(solver_class):
    """Hook called in `test_solver_install`.

    If one solver needs to be skip/xfailed on some
    particular architecture, call pytest.xfail when
    detecting the situation.
    """
    cuda_version = get_cuda_version()
    if solver_class.name.lower() == "cuml":
        if sys.platform == "darwin":
            pytest.xfail("Cuml is not supported on MacOS.")
        if cuda_version is None:
            pytest.xfail("Cuml needs a working GPU hardware.")


def check_test_solver(solver_class):
    is_platform_macOS = sys.platform == "darwin"

    if is_platform_macOS and ('snapml' in solver_class.name.lower()):
        pytest.skip("running snapml on MacOS takes a lot of time.")
