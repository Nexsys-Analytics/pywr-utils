"""UNIT tests for pywr_utils.model_creation — SyntheticModelCreator and the create_synthetic_model helper, exercised in isolation with no pywr, pandas or filesystem-heavy dependencies.

The class is pure data construction: given an input count and a transfer count it emits a PyWr model dictionary. Every assertion below derives its expected value from those two counts rather than hardcoding a total, so the tests stay honest if the node layout changes shape.
"""
import json

import pytest

from pywr_utils.model_creation import SyntheticModelCreator, create_synthetic_model

pytestmark = pytest.mark.unit

# One input produces three nodes: an input, a link and a demand. Each transfer adds one further link node.
NODES_PER_INPUT = 3

# Each input contributes an input->link edge and a link->demand edge; each transfer contributes a link->transfer edge and a transfer->link edge.
EDGES_PER_INPUT = 2
EDGES_PER_TRANSFER = 2


@pytest.mark.parametrize("inputs,transfers", [(1, 0), (3, 2), (5, 12)])
def test_build_model_node_and_edge_counts_follow_from_inputs_and_transfers(inputs, transfers):
    model = SyntheticModelCreator(inputs, transfers).build_model()

    assert len(model["nodes"]) == inputs * NODES_PER_INPUT + transfers
    assert len(model["edges"]) == inputs * EDGES_PER_INPUT + transfers * EDGES_PER_TRANSFER


def test_build_model_edges_only_reference_declared_nodes():
    model = SyntheticModelCreator(4, 6).build_model()
    node_names = {node["name"] for node in model["nodes"]}

    for source, target in model["edges"]:
        assert source in node_names
        assert target in node_names


def test_transfers_wrap_around_when_they_outnumber_inputs():
    """A transfer connects link i to link i+1, cycling back to the first link once it runs off the end — so more transfers than inputs must still produce valid edges rather than an index error."""
    inputs = 3
    model = SyntheticModelCreator(inputs, inputs + 1).build_model()

    wrapping_transfer = f"transfer_{inputs + 1}"
    feeds_the_transfer = [source for source, target in model["edges"] if target == wrapping_transfer]
    fed_by_the_transfer = [target for source, target in model["edges"] if source == wrapping_transfer]

    # The (inputs + 1)th transfer has no link_{inputs + 1} to start from, so it wraps back to the first link and feeds the second.
    assert feeds_the_transfer == ["link_1"]
    assert fed_by_the_transfer == ["link_2"]


def test_create_transfer_nodes_divides_by_input_count_and_rejects_zero_inputs():
    """Transfer routing is modulo the input count, so zero inputs is not a degenerate-but-valid model — it is an error, and it must surface as one rather than silently producing an empty model."""
    creator = SyntheticModelCreator(0, 1)

    with pytest.raises(ZeroDivisionError):
        creator.build_model()


def test_get_model_summary_reports_the_counts_of_the_model_it_describes():
    inputs, transfers = 4, 7
    creator = SyntheticModelCreator(inputs, transfers)
    model = creator.build_model()

    summary = creator.get_model_summary()

    assert f"inputs: {inputs}" in summary
    assert f"Transfers: {transfers}" in summary
    assert f"Total Nodes: {len(model['nodes'])}" in summary
    assert f"Total Edges: {len(model['edges'])}" in summary
    assert f"Parameters: {len(model['parameters'])}" in summary
    assert model["timestepper"]["start"] in summary
    assert model["timestepper"]["end"] in summary


def test_save_model_writes_json_that_round_trips_to_the_built_model(tmp_path):
    creator = SyntheticModelCreator(2, 2)
    destination = tmp_path / "model.json"

    returned_path = creator.save_model(str(destination))

    assert returned_path == str(destination)
    assert json.loads(destination.read_text()) == creator.build_model()


def test_create_synthetic_model_returns_the_model_without_writing_when_no_output_given(tmp_path):
    model = create_synthetic_model(3, 1)

    assert model == SyntheticModelCreator(3, 1).build_model()
    assert list(tmp_path.iterdir()) == []


def test_create_synthetic_model_writes_the_returned_model_when_an_output_is_given(tmp_path):
    destination = tmp_path / "written.json"

    model = create_synthetic_model(3, 1, output_file=str(destination))

    assert json.loads(destination.read_text()) == model
