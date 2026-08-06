"""E2E tests for the pywr-utils command-line interface.

These drive the CLI the way a user does — through the Click entrypoint with argv-style arguments — rather than calling the underlying functions. The final test in this file goes one step further and shells out to the installed `pywr-utils` console script, which is the only way to prove the packaging metadata in pyproject.toml actually produces a working command.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from pywr_utils import model_runner
from pywr_utils.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_help_lists_both_commands(runner):
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "create-synthetic-model" in result.output
    assert "run-incremental-sizes" in result.output


def test_version_is_reported(runner):
    result = runner.invoke(main, ["--version"])

    assert result.exit_code == 0
    assert "pywr-utils" in result.output


def test_create_synthetic_model_writes_the_requested_model(runner, workdir):
    output = workdir / "model.json"

    result = runner.invoke(main, ["create-synthetic-model", "--zones", "3", "--transfers", "2", "--output", str(output)])

    assert result.exit_code == 0, result.output
    model = json.loads(output.read_text())
    assert f"Total nodes: {len(model['nodes'])}" in result.output
    assert f"Total edges: {len(model['edges'])}" in result.output
    assert f"Parameters: {len(model['parameters'])}" in result.output


def test_short_option_names_are_accepted(runner, workdir):
    output = workdir / "short.json"

    result = runner.invoke(main, ["create-synthetic-model", "-z", "2", "-t", "1", "-o", str(output)])

    assert result.exit_code == 0, result.output
    assert output.exists()


def test_output_defaults_to_a_models_directory_named_after_the_size(runner, workdir):
    result = runner.invoke(main, ["create-synthetic-model", "--zones", "3", "--transfers", "2"])

    assert result.exit_code == 0, result.output
    expected = workdir / "models" / "synthetic_model_3z_2t.json"
    assert expected.exists(), f"expected the default path to be created, got {list(workdir.rglob('*.json'))}"


def test_a_missing_output_directory_is_created(runner, workdir):
    output = workdir / "nested" / "deeper" / "model.json"

    result = runner.invoke(main, ["create-synthetic-model", "-z", "2", "-t", "1", "-o", str(output)])

    assert result.exit_code == 0, result.output
    assert output.exists()


def test_show_summary_prints_the_model_summary(runner, workdir):
    output = workdir / "model.json"
    args = ["create-synthetic-model", "-z", "4", "-t", "3", "-o", str(output)]

    without_summary = runner.invoke(main, args)
    with_summary = runner.invoke(main, args + ["--show-summary"])

    assert with_summary.exit_code == 0, with_summary.output
    assert "Synthetic PYWR Model Summary:" in with_summary.output
    assert "Synthetic PYWR Model Summary:" not in without_summary.output
    assert "Total Nodes:" in with_summary.output


def test_the_run_flag_solves_the_model_it_just_created(runner, workdir):
    output = workdir / "model.json"

    result = runner.invoke(main, ["create-synthetic-model", "-z", "3", "-t", "2", "-o", str(output), "--run"])

    assert result.exit_code == 0, result.output
    assert (workdir / f"output_{output.stem}.csv").exists(), "the --run flag should leave the solver's results behind"


def test_an_unwritable_output_path_aborts_with_an_error(runner, workdir):
    blocker = workdir / "blocker"
    blocker.write_text("this is a file, not a directory")
    output = blocker / "model.json"

    result = runner.invoke(main, ["create-synthetic-model", "-z", "2", "-t", "1", "-o", str(output)])

    assert result.exit_code != 0
    assert "Error creating model" in result.output


def test_run_incremental_sizes_sweeps_and_saves_results(runner, workdir):
    result = runner.invoke(
        main,
        ["run-incremental-sizes", "--max-zones", "4", "--zone-increment", "2", "--transfer-increment", "2", "--output", "timings.csv"],
    )

    assert result.exit_code == 0, result.output
    assert "Max zones: 4" in result.output
    assert "completed successfully" in result.output
    saved = (workdir / "timings.csv").read_text()
    assert saved.splitlines()[0].split(",") == ["zones", "transfers", "setup_time", "run_time", "total_time", "error"]


def test_run_incremental_sizes_echoes_the_zone_count_when_max_transfers_is_omitted(runner, workdir):
    result = runner.invoke(
        main,
        ["run-incremental-sizes", "--max-zones", "2", "--zone-increment", "2", "--transfer-increment", "2", "--output", "timings.csv"],
    )

    assert result.exit_code == 0, result.output
    assert "Max transfers: 2" in result.output


def test_run_incremental_sizes_echoes_an_explicit_max_transfers(runner, workdir):
    result = runner.invoke(
        main,
        [
            "run-incremental-sizes",
            "--max-zones", "2",
            "--zone-increment", "2",
            "--max-transfers", "4",
            "--transfer-increment", "2",
            "--output", "timings.csv",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Max transfers: 4" in result.output


def test_show_graph_draws_the_chart(runner, workdir):
    result = runner.invoke(
        main,
        ["run-incremental-sizes", "--max-zones", "2", "--zone-increment", "2", "--transfer-increment", "2", "--output", "timings.csv", "--show-graph"],
    )

    assert result.exit_code == 0, result.output
    assert "SETUP TIME vs TRANSFERS" in result.output


def test_a_failing_sweep_aborts_with_an_error(runner, workdir, monkeypatch):
    failure = "sweep exploded"

    def explode(**kwargs):
        raise RuntimeError(failure)

    monkeypatch.setattr(model_runner, "run_incremental_sizes", explode)

    result = runner.invoke(main, ["run-incremental-sizes", "--max-zones", "2", "--zone-increment", "2", "--output", "timings.csv"])

    assert result.exit_code != 0
    assert failure in result.output


def test_the_installed_console_script_creates_a_model(tmp_path):
    """The truest end-to-end check available: run the `pywr-utils` command that pyproject.toml's [project.scripts] entry installs, in a real subprocess, and confirm it produces a model file."""
    # Resolve the script next to the interpreter running the tests rather than trusting PATH, which does not include the virtualenv's bin directory unless it has been activated.
    alongside_interpreter = Path(sys.executable).parent / "pywr-utils"
    executable = str(alongside_interpreter) if alongside_interpreter.exists() else shutil.which("pywr-utils")
    assert executable, "pywr-utils console script was not found; the package is not installed"
    output = tmp_path / "model.json"

    completed = subprocess.run(
        [executable, "create-synthetic-model", "--zones", "3", "--transfers", "2", "--output", str(output), "--show-summary"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Synthetic PYWR Model Summary:" in completed.stdout
    model = json.loads(output.read_text())
    assert {node["name"] for node in model["nodes"]} >= {"input_1", "link_1", "demand_1", "transfer_1"}
