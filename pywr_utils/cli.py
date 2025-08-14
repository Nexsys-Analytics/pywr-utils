"""Command-line interface for pywr-utils."""

import click
import os
from .model_creation import SyntheticModelCreator
from .model_runner import run_file, run_incremental_sizes


@click.group()
@click.version_option(version="0.1.0", prog_name="pywr-utils")
def main():
    """PYWR utilities command-line interface."""
    pass


@main.command("create-synthetic-model")
@click.option(
    "--zones",
    "-z",
    type=int,
    default=10,
    help="Number of zones to create in the synthetic model (default: 10)"
)
@click.option(
    "--transfers",
    "-t",
    type=int,
    default=10,
    help="Number of transfers to create in the synthetic model (default: 10)"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output filename for the model (defaults to synthetic_model_<zones>z_<transfers>t.json)"
)
@click.option(
    "--show-summary",
    "-s",
    is_flag=True,
    help="Show a summary of the created model"
)
@click.option(
    "--run",
    "-r",
    is_flag=True,
    help="Run the model after creation"
)
def create_synthetic_model(zones, transfers, output, show_summary, run):
    """Create a synthetic PYWR model with specified zones and transfers."""
    click.echo(f"Creating synthetic model with {zones} zones and {transfers} transfers...")
    
    # Create the model
    creator = SyntheticModelCreator(zones, transfers)
    
    # Generate default filename if not provided
    if not output:
        output = os.path.join("models", f"synthetic_model_{zones}z_{transfers}t.json")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Save the model
        saved_file = creator.save_model(output)
        click.echo(f"✓ Model saved to: {saved_file}")
        
        # Show summary if requested
        if show_summary:
            click.echo("\n" + creator.get_model_summary())
        
        # Show basic statistics
        model = creator.build_model()
        click.echo(f"\nModel Statistics:")
        click.echo(f"  • Total nodes: {len(model['nodes'])}")
        click.echo(f"  • Total edges: {len(model['edges'])}")
        click.echo(f"  • Parameters: {len(model['parameters'])}")
        
        click.echo(f"\n✓ Synthetic model creation completed successfully!")
        
        # Run the model if requested
        if run:
            click.echo(f"\nRunning model...")
            run_file(saved_file)
        
    except Exception as e:
        click.echo(f"✗ Error creating model: {str(e)}", err=True)
        raise click.Abort()


@main.command("run-incremental-sizes")
@click.option(
    "--max-zones",
    type=int,
    default=100,
    help="Maximum number of zones to test (default: 100)"
)
@click.option(
    "--zone-increment",
    type=int,
    default=10,
    help="Increment for zones (default: 10)"
)
@click.option(
    "--max-transfers",
    type=int,
    default=None,
    help="Maximum number of transfers per zone size (defaults to max-zones)"
)
@click.option(
    "--transfer-increment",
    type=int,
    default=10,
    help="Increment for transfers (default: 10)"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="incremental_timing_results.csv",
    help="Output CSV filename for results (default: incremental_timing_results.csv)"
)
@click.option(
    "--show-graph",
    "-g",
    is_flag=True,
    help="Display a terminal graph of setup time vs transfers"
)
def run_incremental_sizes_cmd(max_zones, zone_increment, max_transfers, transfer_increment, output, show_graph):
    """Run incremental model sizes and measure setup times for performance analysis."""
    click.echo(f"Running incremental size tests...")
    click.echo(f"  Max zones: {max_zones}")
    click.echo(f"  Zone increment: {zone_increment}")
    click.echo(f"  Max transfers: {max_transfers if max_transfers else max_zones}")
    click.echo(f"  Transfer increment: {transfer_increment}")
    click.echo(f"  Output file: {output}")
    
    try:
        results_df = run_incremental_sizes(
            max_zones=max_zones,
            zone_increment=zone_increment,
            max_transfers=max_transfers,
            transfer_increment=transfer_increment,
            output_csv=output,
            show_graph=show_graph
        )
        
        click.echo(f"\n✓ Incremental size testing completed successfully!")
        click.echo(f"Results saved to: {output}")

    except Exception as e:
        click.echo(f"✗ Error running incremental sizes: {str(e)}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
