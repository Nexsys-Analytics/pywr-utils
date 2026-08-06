"""UNIT tests for the terminal-plotting helpers in pywr_utils.model_runner — _display_terminal_graph and _draw_line.

Both are pure functions over an in-memory DataFrame or grid: no PyWr model, no solver, no filesystem. They are tested here in isolation from the run machinery that normally feeds them.
"""
import pandas as pd
import pytest

from pywr_utils.model_runner import _display_terminal_graph, _draw_line


def timing_frame(zones, transfers, setup_times):
    return pd.DataFrame({"zones": zones, "transfers": transfers, "setup_time": setup_times})


def test_empty_frame_reports_no_data_rather_than_raising(capsys):
    _display_terminal_graph(pd.DataFrame(columns=["zones", "transfers", "setup_time"]))

    assert "No data to plot." in capsys.readouterr().out


# Each dataset is a permutation of the same four setup times against the same four transfer counts, chosen so the Pearson correlation is an exact, hand-checkable value that lands squarely inside one classification band.
@pytest.mark.parametrize(
    "setup_times,expected_correlation,expected_label",
    [
        ([1.0, 2.0, 3.0, 4.0], 1.0, "Strong positive correlation"),
        ([2.0, 1.0, 4.0, 3.0], 0.6, "Moderate positive correlation"),
        ([2.0, 4.0, 1.0, 3.0], 0.0, "Weak correlation"),
        ([3.0, 4.0, 1.0, 2.0], -0.6, "Moderate negative correlation"),
        ([4.0, 3.0, 2.0, 1.0], -1.0, "Strong negative correlation"),
    ],
)
def test_overall_correlation_is_reported_and_classified(capsys, setup_times, expected_correlation, expected_label):
    transfers = [1, 2, 3, 4]
    frame = timing_frame([10] * len(transfers), transfers, setup_times)

    assert frame["transfers"].corr(frame["setup_time"]) == pytest.approx(expected_correlation)

    _display_terminal_graph(frame)
    output = capsys.readouterr().out

    assert f"Overall correlation: {expected_correlation:.3f}" in output
    assert expected_label in output


def test_single_zone_count_omits_the_per_zone_correlation_breakdown(capsys):
    _display_terminal_graph(timing_frame([10, 10], [1, 2], [0.1, 0.2]))

    assert "Correlations by zone count:" not in capsys.readouterr().out


def test_each_zone_count_gets_its_own_legend_marker_and_correlation(capsys):
    frame = timing_frame(
        zones=[10, 10, 10, 20, 20, 20],
        transfers=[1, 2, 3, 1, 2, 3],
        setup_times=[0.1, 0.2, 0.3, 0.5, 0.4, 0.6],
    )

    _display_terminal_graph(frame)
    output = capsys.readouterr().out

    legend_markers = [line.split("=")[0].strip() for line in output.splitlines() if line.strip().endswith("zones")]
    assert len(legend_markers) == frame["zones"].nunique()
    assert len(set(legend_markers)) == len(legend_markers), "each zone count needs a distinct marker"

    assert "Correlations by zone count:" in output
    for zones, group in frame.groupby("zones"):
        expected = group["transfers"].corr(group["setup_time"])
        assert f"{zones} zones: {expected:.3f}" in output

    for marker in legend_markers:
        assert marker in output, "every legend marker should also appear plotted on the chart"


def test_reported_ranges_come_from_the_data(capsys):
    frame = timing_frame([10, 10, 20], [5, 40, 25], [0.25, 1.5, 0.75])

    _display_terminal_graph(frame)
    output = capsys.readouterr().out

    assert "Transfer range: 5 - 40" in output
    assert "Setup time range: 0.250s - 1.500s" in output
    assert f"{len(frame)} models tested" in output


def test_single_point_is_plotted_without_a_correlation(capsys):
    """One row means no range to scale against and no correlation to compute; the chart must still render rather than dividing by a zero span."""
    _display_terminal_graph(timing_frame([10], [7], [0.4]))
    output = capsys.readouterr().out

    assert "Transfer range: 7 - 7" in output
    assert "Overall correlation" not in output


def test_constant_setup_times_suppress_the_undefined_correlation(capsys):
    """Zero variance in setup_time makes the Pearson correlation NaN — printing 'nan' would be worse than printing nothing."""
    frame = timing_frame([10, 10, 10], [1, 2, 3], [0.5, 0.5, 0.5])

    assert pd.isna(frame["transfers"].corr(frame["setup_time"]))

    _display_terminal_graph(frame)
    output = capsys.readouterr().out

    assert "Overall correlation" not in output
    assert "nan" not in output.lower()


def blank_grid(width, height):
    return [[" " for _ in range(width)] for _ in range(height)]


def test_draw_line_marks_a_diagonal_path_between_the_endpoints():
    width = height = 5
    grid = blank_grid(width, height)

    _draw_line(grid, 0, 0, 4, 4, "●", width, height)

    for i in range(5):
        assert grid[i][i] == "·", f"expected the diagonal cell ({i}, {i}) to be on the drawn path"
    assert grid[0][4] == " ", "cells off the path must be left blank"


def test_draw_line_marks_a_horizontal_path():
    width, height = 6, 3
    grid = blank_grid(width, height)

    _draw_line(grid, 1, 1, 4, 1, "●", width, height)

    assert "".join(grid[1]) == " ···· "
    assert "".join(grid[0]).strip() == ""


def test_draw_line_does_not_overwrite_existing_plotted_points():
    """Line segments are connective tissue between real data points — a segment must never erase the marker of a point it passes through."""
    width = height = 5
    grid = blank_grid(width, height)
    grid[2][2] = "●"

    _draw_line(grid, 0, 0, 4, 4, "■", width, height)

    assert grid[2][2] == "●"


def test_draw_line_clips_to_the_grid_instead_of_writing_out_of_bounds():
    """Negative coordinates must be skipped, not passed through to the list — Python would silently index from the end of the row and paint a segment on the wrong side of the chart."""
    width = height = 4
    grid = blank_grid(width, height)

    _draw_line(grid, 0, -3, 0, 1, "●", width, height)

    marked = {(x, y) for y in range(height) for x in range(width) if grid[y][x] != " "}
    assert marked == {(0, 0), (0, 1)}


def test_draw_line_between_identical_points_marks_only_that_point():
    width = height = 3
    grid = blank_grid(width, height)

    _draw_line(grid, 1, 1, 1, 1, "●", width, height)

    assert grid[1][1] == "·"
    assert sum(cell != " " for row in grid for cell in row) == 1
