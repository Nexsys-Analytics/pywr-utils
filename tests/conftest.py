import sys

import pytest

BLOCKED_MODULES = ("pandas", "pywr")


@pytest.fixture
def block_pandas_and_pywr(monkeypatch):
    """Simulate a fresh install with neither pandas nor pywr present.

    Setting `sys.modules[name] = None` is the standard idiom for this: Python's import system raises ImportError for any name mapped to None in sys.modules, without needing to intercept `__import__` or actually uninstall the package.
    """
    for name in BLOCKED_MODULES:
        monkeypatch.setitem(sys.modules, name, None)
    for name in ("pywr_utils.cli", "pywr_utils.model_runner"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield
    for name in ("pywr_utils.cli", "pywr_utils.model_runner"):
        monkeypatch.delitem(sys.modules, name, raising=False)
