"""Regression tests for the model_runner lazy-import fix.

create-synthetic-model must not require pandas or pywr to be installed, since it only depends on model_creation (stdlib json/typing). Before the fix, cli.py imported model_runner (which needs both) at module load time, so every CLI invocation failed on a fresh install before pywr's own native build was ever reached.
"""
import importlib
import sys

import pytest
from click.testing import CliRunner

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


def test_cli_module_imports_without_pandas_or_pywr(block_pandas_and_pywr):
    module = importlib.import_module("pywr_utils.cli")
    assert hasattr(module, "create_synthetic_model")


def test_create_synthetic_model_command_without_pandas_or_pywr(tmp_path, block_pandas_and_pywr):
    cli = importlib.import_module("pywr_utils.cli")
    output_path = tmp_path / "synthetic.json"

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["create-synthetic-model", "--zones", "2", "--transfers", "1", "--output", str(output_path)],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_create_synthetic_model_command_run_flag_requires_pywr(tmp_path, block_pandas_and_pywr):
    """--run does need pywr, and must fail with the real ImportError rather than something masking it — the lazy import only defers the cost, it doesn't remove the dependency for the code path that actually uses it."""
    cli = importlib.import_module("pywr_utils.cli")
    output_path = tmp_path / "synthetic.json"

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["create-synthetic-model", "--zones", "2", "--transfers", "1", "--output", str(output_path), "--run"],
    )

    assert result.exit_code != 0
    assert output_path.exists(), "model should still be saved before the run step is attempted"
