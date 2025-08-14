"""Model creation utilities for PYWR synthetic models."""

import json
from typing import Dict, Any, List, Optional


class SyntheticModelCreator:
    """Creates synthetic PYWR models with specified inputs and transfers."""
    
    def __init__(self, inputs: int, transfers: int):
        """Initialize the model creator.
        
        Args:
            inputs: Number of inputs to create
            transfers: Number of transfers to create
        """
        self.inputs = inputs
        self.transfers = transfers
        self.model_data = {
            "metadata": {
                "title": f"Synthetic Model - {inputs} inputs, {transfers} Transfers",
                "description": "Auto-generated synthetic PYWR model",
                "minimum_version": "1.0"
            },
            "timestepper": {
                "start": "2020-01-01",
                "end": "2020-12-31",
                "timestep": 1
            },
            "nodes": [],
            "edges": [],
            "parameters": []
        }
    
    def create_inputs(self) -> List[Dict[str, Any]]:
        """Create input nodes for the model.
        
        Returns:
            List of input node definitions
        """
        inputs = []
        for i in range(self.inputs):
            input = {
                "name": f"input_{i+1}",
                "type": "input",
                "max_flow": {
                    "type": "constant",
                    "value": 10.0 + i * 5  # Varying flow for each input
                },
                "position": {
                    "geographic": [0.0 + i * 0.1, 51.0 + i * 0.1]
                }
            }
            inputs.append(input)
        return inputs
    
    def create_link_nodes(self) -> List[Dict[str, Any]]:
        """Create link nodes for each input.

        Returns:
            List of link node definitions
        """
        link_nodes = []
        for i in range(self.inputs):
            link = {
                "name": f"link_{i+1}",
                "type": "link",
                "position": {
                    "geographic": [0.05 + i * 0.1, 51.05 + i * 0.1]
                }
            }
            link_nodes.append(link)
        return link_nodes

    def create_demand_nodes(self) -> List[Dict[str, Any]]:
        """Create demand nodes for the model.
        
        Returns:
            List of demand node definitions
        """
        demand_nodes = []
        for i in range(self.inputs):
            demand = {
                "name": f"demand_{i+1}",
                "type": "output",
                "max_flow": 8.0 + i * 2,  # Varying demand
                "cost": -10.0,  # Negative cost to prioritize meeting demand
                "position": {
                    "geographic": [0.1 + i * 0.1, 51.1 + i * 0.1]
                }
            }
            demand_nodes.append(demand)
        return demand_nodes
    
    def create_transfer_nodes(self) -> List[Dict[str, Any]]:
        """Create transfer link nodes.
        
        Returns:
            List of transfer node definitions
        """
        transfer_nodes = []
        for i in range(self.transfers):
            # Create transfers between adjacent inputs (cycling if needed)
            from_input = i % self.inputs
            to_input = (i + 1) % self.inputs
            
            transfer = {
                "name": f"transfer_{i+1}",
                "type": "link",
                "max_flow": 15.0,
                "cost": 1.0 + i * 0.5,  # Varying transfer costs
                "position": {
                    "geographic": [0.075 + from_input * 0.1, 51.075 + to_input * 0.1]
                }
            }
            transfer_nodes.append(transfer)
        return transfer_nodes
    
    def create_edges(self) -> List[List[str]]:
        """Create edges connecting the nodes.
        
        Returns:
            List of edge definitions [from_node, to_node]
        """
        edges = []

        # Connect inputs to the links
        for i in range(self.inputs):
            edges.append([f"input_{i+1}", f"link_{i+1}"])

        # Connect links to demands
        for i in range(self.inputs):
            edges.append([f"link_{i+1}", f"demand_{i+1}"])

        # Connect transfer links
        for i in range(self.transfers):
            from_input = i % self.inputs
            to_input = (i + 1) % self.inputs
            # From link to transfer
            edges.append([f"link_{from_input+1}", f"transfer_{i+1}"])
            # From transfer to destination link
            edges.append([f"transfer_{i+1}", f"link_{to_input+1}"])

        return edges

    def create_parameters(self) -> Dict[str, Any]:
        """Create model parameters.
        
        Returns:
            List of parameter definitions
        """
        parameters = {}
        
        return parameters
    
    def build_model(self) -> Dict[str, Any]:
        """Build the complete model structure.
        
        Returns:
            Complete PYWR model dictionary
        """
        # Create all node types
        inputs = self.create_inputs()
        link_nodes = self.create_link_nodes()
        demand_nodes = self.create_demand_nodes()
        transfer_nodes = self.create_transfer_nodes()
        
        # Combine all nodes
        all_nodes = inputs + link_nodes + demand_nodes + transfer_nodes
        self.model_data["nodes"] = all_nodes
        
        # Create edges
        self.model_data["edges"] = self.create_edges()
        
        # Create parameters
        self.model_data["parameters"] = self.create_parameters()
        
        return self.model_data
    
    def save_model(self, filename: str) -> str:
        """Save the model to a JSON file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to the saved file
        """
        model = self.build_model()
        
        with open(filename, 'w') as f:
            json.dump(model, f, indent=2)
        
        return filename
    
    def get_model_summary(self) -> str:
        """Get a summary of the created model.
        
        Returns:
            String summary of the model
        """
        model = self.build_model()
        
        summary_lines = [
            f"Synthetic PYWR Model Summary:",
            f"  inputs: {self.inputs}",
            f"  Transfers: {self.transfers}",
            f"  Total Nodes: {len(model['nodes'])}",
            f"    - Catchments: {self.inputs}",
            f"    - Reservoirs: {self.inputs}",
            f"    - Demands: {self.inputs}",
            f"    - Transfers: {self.transfers}",
            f"  Total Edges: {len(model['edges'])}",
            f"  Parameters: {len(model['parameters'])}",
            f"  Time Period: {model['timestepper']['start']} to {model['timestepper']['end']}"
        ]
        
        return "\n".join(summary_lines)


def create_synthetic_model(inputs: int, transfers: int, output_file: Optional[str] = None) -> Dict[str, Any]:
    """Create a synthetic PYWR model.
    
    Args:
        inputs: Number of inputs to create
        transfers: Number of transfers to create
        output_file: Optional output filename
        
    Returns:
        Model dictionary
    """
    creator = SyntheticModelCreator(inputs, transfers)
    model = creator.build_model()
    
    if output_file:
        creator.save_model(output_file)
    
    return model
