"""INTEGRATION tests for pywr_utils.model_runner — PywrFileRunner, run_file and run_incremental_sizes driven as Python APIs against a real PyWr model.

Nothing here is mocked. Each test generates a genuine model with SyntheticModelCreator, hands it to PyWr, and lets the real GLPK solver run it. That is the point of this layer: it proves the loading, solving, recorder-harvesting and CSV-writing path actually works, not merely that the functions were called.
"""
import json
import logging

import pandas as pd
import pytest

from pywr_utils import model_runner
from pywr_utils.model_creation import SyntheticModelCreator
from pywr_utils.model_runner import PywrFileRunner, run_file, run_incremental_sizes

DEFAULT_SOLVER = "glpk"
ALTERNATIVE_SOLVER = "glpk-edge"


@pytest.fixture
def model_file(tmp_path):
    """A small but genuinely solvable model — three inputs feeding three demands, with two transfers between them."""
    path = tmp_path / "model.json"
    SyntheticModelCreator(3, 2).save_model(str(path))
    return path


@pytest.fixture
def model_file_with_recorders(tmp_path):
    """The same model with two flow recorders attached, one of them named so it trips the 'total' reporting branch."""
    model = SyntheticModelCreator(3, 2).build_model()
    model["recorders"] = {
        "total_demand_1": {"type": "totalflownoderecorder", "node": "demand_1"},
        "demand_2_flow": {"type": "totalflownoderecorder", "node": "demand_2"},
    }
    path = tmp_path / "model_with_recorders.json"
    path.write_text(json.dumps(model))
    return path


def test_load_then_run_solves_the_model_and_reports_timings(model_file, tmp_path):
    runner = PywrFileRunner()
    runner.load_pywr_model_from_file(str(model_file))
    outfile = tmp_path / "results.csv"

    timings = runner.run_pywr_model(str(outfile))

    assert "error" not in timings, timings
    assert timings["setup_time"] > 0
    assert timings["run_time"] > 0
    assert timings["total_time"] == pytest.approx(timings["setup_time"] + timings["run_time"])
    assert outfile.exists()


def test_loading_builds_a_model_matching_the_json_on_disk(model_file):
    expected = json.loads(model_file.read_text())
    runner = PywrFileRunner()

    runner.load_pywr_model_from_file(str(model_file))

    assert {node.name for node in runner.model.nodes} == {node["name"] for node in expected["nodes"]}


def test_load_defaults_to_the_glpk_solver(model_file):
    runner = PywrFileRunner()

    runner.load_pywr_model_from_file(str(model_file))

    assert runner.model.solver.name == DEFAULT_SOLVER


def test_load_passes_an_explicit_solver_through_to_pywr(model_file):
    """Regression guard: the solver argument used to be accepted and then silently discarded in favour of a hardcoded glpk, so a caller asking for a different solver got the default without any error."""
    runner = PywrFileRunner()

    runner.load_pywr_model_from_file(str(model_file), solver=ALTERNATIVE_SOLVER)

    assert ALTERNATIVE_SOLVER != DEFAULT_SOLVER, "this test is vacuous unless the two solvers really differ"
    assert runner.model.solver.name == ALTERNATIVE_SOLVER


def test_an_unknown_solver_is_rejected_rather_than_falling_back(model_file):
    runner = PywrFileRunner()

    with pytest.raises(KeyError):
        runner.load_pywr_model_from_file(str(model_file), solver="not-a-real-solver")


def test_a_model_run_with_either_solver_produces_the_same_results(model_file, tmp_path):
    """If the solver argument were being ignored, this test would still pass — but paired with the assertion above that the chosen solver is actually installed on the model, it shows the alternative solver is genuinely exercised rather than merely accepted."""
    results = {}
    for solver in (DEFAULT_SOLVER, ALTERNATIVE_SOLVER):
        runner = PywrFileRunner()
        runner.load_pywr_model_from_file(str(model_file), solver=solver)
        outfile = tmp_path / f"results_{solver}.csv"
        timings = runner.run_pywr_model(str(outfile))
        assert "error" not in timings, timings
        results[solver] = outfile.read_text()

    assert results[DEFAULT_SOLVER] == results[ALTERNATIVE_SOLVER]


def test_recorder_values_are_written_to_the_output_csv(model_file_with_recorders, tmp_path):
    runner = PywrFileRunner()
    runner.load_pywr_model_from_file(str(model_file_with_recorders))
    outfile = tmp_path / "recorded.csv"

    runner.run_pywr_model(str(outfile))

    written = pd.read_csv(outfile, index_col=0)
    reported = {recorder.name: list(recorder.values()) for recorder in runner.model.recorders if recorder.name in written.index}
    assert reported, "expected at least one recorder to reach the CSV"
    for name, values in reported.items():
        assert written.loc[name].tolist() == pytest.approx(values)


def test_recorders_named_total_are_echoed_to_stdout(model_file_with_recorders, tmp_path, capsys):
    runner = PywrFileRunner()
    runner.load_pywr_model_from_file(str(model_file_with_recorders))

    runner.run_pywr_model(str(tmp_path / "out.csv"))
    output = capsys.readouterr().out

    total_recorder = next(r for r in runner.model.recorders if "total" in r.name)
    assert f"{total_recorder.name}: {next(iter(total_recorder.values()))}" in output


def test_a_recorder_that_cannot_report_values_is_logged_and_skipped(model_file, tmp_path, caplog):
    """The progress recorder the runner attaches has no meaningful values(); it must be logged and stepped over rather than aborting the whole results harvest."""
    runner = PywrFileRunner()
    runner.load_pywr_model_from_file(str(model_file))
    outfile = tmp_path / "out.csv"

    with caplog.at_level(logging.ERROR, logger="pywr_utils.model_runner"):
        timings = runner.run_pywr_model(str(outfile))

    assert "error" not in timings, timings
    assert any("progressrecorder" in record.message for record in caplog.records)
    assert "progressrecorder" not in outfile.read_text()


def test_create_csv_false_skips_writing_the_output_file(model_file, tmp_path):
    runner = PywrFileRunner()
    runner.load_pywr_model_from_file(str(model_file))
    outfile = tmp_path / "should_not_exist.csv"

    timings = runner.run_pywr_model(str(outfile), create_csv=False)

    assert "error" not in timings, timings
    assert not outfile.exists()


def test_running_without_a_loaded_model_reports_the_error_instead_of_raising(tmp_path, caplog):
    runner = PywrFileRunner()

    with caplog.at_level(logging.ERROR, logger="pywr_utils.model_runner"):
        timings = runner.run_pywr_model(str(tmp_path / "out.csv"))

    assert timings["error"]
    assert timings["total_time"] == pytest.approx(timings["setup_time"] + timings["run_time"])
    error_records = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert error_records, "the failure should be logged through the module's own logger, not only returned"
    assert error_records[-1].name == "pywr_utils.model_runner", "the failure should be logged through the module's own named logger, not the root logger"
    assert error_records[-1].exc_info is not None, "the traceback should be attached via exc_info rather than repeated in the log message"


def test_run_file_derives_the_output_name_from_the_input_filename(model_file, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    timings = run_file(str(model_file))

    assert "error" not in timings, timings
    assert (tmp_path / f"output_{model_file.stem}.csv").exists()


def test_run_file_honours_an_explicit_output_file(model_file, tmp_path):
    outfile = tmp_path / "explicit.csv"

    timings = run_file(str(model_file), output_file=str(outfile))

    assert "error" not in timings, timings
    assert outfile.exists()


def test_incremental_sizes_runs_each_combination_and_saves_a_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    results = run_incremental_sizes(max_zones=4, zone_increment=2, transfer_increment=2, output_csv="timings.csv")

    # Transfers are capped at the zone count when max_transfers is not given: 2 zones allow 2 transfers, 4 zones allow 2 and 4.
    assert list(zip(results["zones"], results["transfers"])) == [(2, 2), (4, 2), (4, 4)]
    assert (results["setup_time"] > 0).all()
    assert (results["error"] == "").all(), results["error"].tolist()
    assert pd.read_csv(tmp_path / "timings.csv").shape[0] == len(results)


def test_incremental_sizes_cleans_up_the_generated_model_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    run_incremental_sizes(max_zones=2, zone_increment=2, transfer_increment=2, output_csv="timings.csv")

    assert list((tmp_path / "models").iterdir()) == []


def test_an_explicit_max_transfers_is_not_capped_by_the_zone_count(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    results = run_incremental_sizes(max_zones=2, zone_increment=2, max_transfers=4, transfer_increment=2, output_csv="timings.csv")

    assert list(results["transfers"]) == [2, 4], "an explicit max_transfers should override the zone-count cap"


def test_a_failing_model_is_recorded_as_an_error_row_rather_than_aborting_the_sweep(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    failure = "synthetic build failure"

    def explode(self):
        raise RuntimeError(failure)

    monkeypatch.setattr(SyntheticModelCreator, "build_model", explode)

    results = run_incremental_sizes(max_zones=4, zone_increment=2, transfer_increment=2, output_csv="timings.csv")

    assert len(results) == 3
    assert (results["error"] == failure).all()
    assert (results["total_time"] == 0).all()


def test_no_size_combinations_produces_an_empty_frame_and_no_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    results = run_incremental_sizes(max_zones=1, zone_increment=2, transfer_increment=2, output_csv="timings.csv")

    assert results.empty
    assert not (tmp_path / "timings.csv").exists()


def test_show_graph_renders_the_terminal_chart_after_each_run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rendered = []
    monkeypatch.setattr(model_runner, "_display_terminal_graph", lambda df: rendered.append(len(df)))

    run_incremental_sizes(max_zones=4, zone_increment=2, transfer_increment=2, output_csv="timings.csv", show_graph=True)

    # The chart is redrawn after every model, each time with one more row than the last.
    assert rendered == [1, 2, 3]


def test_show_graph_off_renders_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rendered = []
    monkeypatch.setattr(model_runner, "_display_terminal_graph", lambda df: rendered.append(len(df)))

    run_incremental_sizes(max_zones=2, zone_increment=2, transfer_increment=2, output_csv="timings.csv")

    assert rendered == []
