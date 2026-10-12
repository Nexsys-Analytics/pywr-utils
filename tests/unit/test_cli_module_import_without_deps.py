"""UNIT test for the model_runner lazy-import fix.

create-synthetic-model must not require pandas or pywr to be installed, since it only depends on model_creation (stdlib json/typing). Before the fix, cli.py imported model_runner (which needs both) at module load time, so every CLI invocation failed on a fresh install before pywr's own native build was ever reached. This checks only the import itself, in isolation from the Click entrypoint — see tests/e2e/test_cli_lazy_import.py for the command-level regression coverage.
"""
import importlib

import pytest

pytestmark = pytest.mark.unit


def test_cli_module_imports_without_pandas_or_pywr(block_pandas_and_pywr):
    module = importlib.import_module("pywr_utils.cli")
    assert hasattr(module, "create_synthetic_model")
