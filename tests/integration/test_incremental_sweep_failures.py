"""INTEGRATION tests for the failure and cleanup paths of run_incremental_sizes.

The sweep runs real models, so a failure part way through has to leave the working directory tidy and still surface a usable results frame. These tests drive the real function and inject failures at the filesystem boundary rather than stubbing the sweep itself.
"""
import os

import pandas as pd
import pytest

from pywr_utils import model_runner
from pywr_utils.model_runner import PywrFileRunner, run_incremental_sizes

SWEEP_ARGS = {"max_zones": 2, "zone_increment": 2, "transfer_increment": 2, "output_csv": "timings.csv"}


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_an_undeletable_model_file_warns_without_failing_the_sweep(workdir, monkeypatch, capsys):
    def refuse_to_delete(path):
        raise OSError("permission denied")

    monkeypatch.setattr(os, "unlink", refuse_to_delete)

    results = run_incremental_sizes(**SWEEP_ARGS)
    output = capsys.readouterr().out

    assert len(results) == 1
    assert (results["error"] == "").all(), "a cleanup failure must not be recorded as a model failure"
    assert "Warning: Could not delete" in output
    assert (workdir / "timings.csv").exists()


def test_a_model_that_fails_after_being_written_is_still_cleaned_up(workdir, capsys, monkeypatch):
    failure = "load exploded"

    def explode(self, filename, solver=None):
        raise RuntimeError(failure)

    monkeypatch.setattr(PywrFileRunner, "load_pywr_model_from_file", explode)

    results = run_incremental_sizes(**SWEEP_ARGS)
    output = capsys.readouterr().out

    assert (results["error"] == failure).all()
    assert "Cleaned up after error" in output
    assert list((workdir / "models").iterdir()) == [], "the half-finished model file must not be left behind"


def test_a_failed_model_whose_file_also_resists_deletion_still_returns_results(workdir, monkeypatch):
    """Both the run and the cleanup fail. The sweep must still hand back the error row rather than letting the cleanup's own exception escape and destroy the results of every model before it."""

    def explode(self, filename, solver=None):
        raise RuntimeError("load exploded")

    def refuse_to_delete(path):
        raise OSError("permission denied")

    monkeypatch.setattr(PywrFileRunner, "load_pywr_model_from_file", explode)
    monkeypatch.setattr(os, "unlink", refuse_to_delete)

    results = run_incremental_sizes(**SWEEP_ARGS)

    assert (results["error"] == "load exploded").all()
    assert (workdir / "timings.csv").exists()


def test_the_chart_is_still_drawn_for_a_failed_model(workdir, monkeypatch):
    rendered = []
    monkeypatch.setattr(model_runner, "_display_terminal_graph", lambda df: rendered.append(len(df)))

    def explode(self, filename, solver=None):
        raise RuntimeError("load exploded")

    monkeypatch.setattr(PywrFileRunner, "load_pywr_model_from_file", explode)

    run_incremental_sizes(show_graph=True, **SWEEP_ARGS)

    assert rendered == [1], "an error row still belongs on the chart"


def test_a_failure_writing_the_summary_is_reported_and_propagated(workdir, monkeypatch, capsys):
    """The sweep is expensive, so a failure to persist its results must be loud rather than swallowed into a silently missing file."""
    failure = "disk full"

    def refuse_to_write(self, *args, **kwargs):
        raise OSError(failure)

    monkeypatch.setattr(pd.DataFrame, "to_csv", refuse_to_write)

    with pytest.raises(OSError, match=failure):
        run_incremental_sizes(**SWEEP_ARGS)

    assert "Error saving CSV" in capsys.readouterr().out


def test_a_summary_that_silently_fails_to_appear_is_flagged(workdir, monkeypatch, capsys):
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda self, *args, **kwargs: None)

    run_incremental_sizes(**SWEEP_ARGS)
    output = capsys.readouterr().out

    assert "Output file was not created" in output
    assert not (workdir / "timings.csv").exists()


def test_the_confirmed_summary_reports_its_size_on_disk(workdir, capsys):
    run_incremental_sizes(**SWEEP_ARGS)
    output = capsys.readouterr().out

    assert "Output file confirmed" in output
    assert f"File size: {(workdir / 'timings.csv').stat().st_size} bytes" in output


@pytest.mark.xfail(
    strict=True,
    reason="run_incremental_sizes joins the output directory onto a path that already contains it, so an output_csv with any directory component is written to <dir>/<dir>/<name> and the whole sweep's results are lost to an OSError after every model has already been run.",
)
def test_an_output_path_with_a_directory_component_is_written_where_it_was_asked_for(workdir):
    run_incremental_sizes(max_zones=2, zone_increment=2, transfer_increment=2, output_csv="results/timings.csv")

    assert (workdir / "results" / "timings.csv").exists()
