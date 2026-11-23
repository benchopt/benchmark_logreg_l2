import os
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


def check_test_dataset_get_data(dataset_class):
    """Hook called in `test_dataset_get_data`."""
    if dataset_class.name.lower() == "criteo":
        pytest.skip("libsvmdata 0.5 no longer provides criteo.")
    # The real datasets download from external servers: not to make the CI of
    # each PR depend on them, they only run in the weekly CI, with --runxfail.
    if os.environ.get("GITHUB_ACTIONS") and dataset_class.name != "Simulated":
        pytest.xfail("Downloads the data, tested in the weekly CI.")
