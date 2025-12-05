import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--ci",
        action="store_true",
        default=False,
        help="Configure tests for continuous integration environment",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--ci"):
        # Filter to keep only tests from test_python_bindings.py
        filtered_items = [item for item in items if "test_python_bindings.py" in item.nodeid]
        items[:] = filtered_items
