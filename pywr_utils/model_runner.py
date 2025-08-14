import json
import os
import pandas as pd
import time
from typing import Dict, Any

from pywr.model import Model
from pywr.recorders.progress import ProgressRecorder

import logging
logging.basicConfig(level='INFO')

def run_file(filepath, output_file=None):
    """Run a PYWR model from file and return timing information."""
    pfr = PywrFileRunner()
    pfr.load_pywr_model_from_file(filepath)
    if not output_file:
        #extract just the filename from the path
        file_name = os.path.basename(filepath)
        output_file = f"output_{file_name.split('.')[0]}.csv"
    timing_info = pfr.run_pywr_model(output_file)
    return timing_info

class PywrFileRunner():
    def __init__(self):
        self.model = None
        self.log = logging.getLogger(__name__)

    def load_pywr_model_from_file(self, filename, solver=None):
        modelpath = os.path.abspath(filename)
        
        with open(modelpath, 'r') as f:
            pywr_data = json.load(f)

        #self.model = Model.load(pywr_data, solver='lpsolve')
        self.model = Model.load(pywr_data, solver='glpk')

    def run_pywr_model(self, outfile="output_1.csv", create_csv=True):
        """Run the PYWR model and return timing information.
        
        Args:
            outfile: Output CSV filename (ignored if create_csv=False)
            create_csv: Whether to create CSV output file (default: True)
        """
        setup_time = 0
        run_time = 0
        
        try:
            # Add a progress recorder to monitor the run.
            ProgressRecorder(self.model)

            print("Setting up the model...")
            # Time the setup phase
            setup_start = time.time()
            # Force a setup regardless of whether the model has been run or setup before
            self.model.setup()
            setup_time = time.time() - setup_start
            print(f"Model setup complete in {setup_time:.3f} seconds. Running the model...")
            
            # Time the run phase
            run_start = time.time()
            run_stats = self.model.run()
            run_time = time.time() - run_start
            self.log.info(run_stats)

            # dataframes_to_output = []
            # columns = []
            columns = []
            data = []
            print("Model run complete. Processing results...")
            for r in self.model.recorders:
                if hasattr(r, 'values'):
                    try:
                        if sum(r.values()) > 0:
                            columns.append(r.name)
                            data.append(list(r.values()))
                    except NotImplementedError as e:
                        self.log.error(f"Error processing recorder {r.name}: {e}")
                    if 'total' in r.name:
                        print(f"{r.name}: {list(r.values())[0]}")

            # Only create CSV if requested
            if create_csv:
                df = pd.DataFrame(data, index=columns)
                df.to_csv(outfile)
                print(f"Output written to {os.path.abspath(outfile)}")

            self.log.info("Model run complete.")
            return {
                'setup_time': setup_time,
                'run_time': run_time,
                'total_time': setup_time + run_time
            }
        except Exception as e:
            logging.exception(e)
            return {
                'setup_time': setup_time,
                'run_time': run_time,
                'total_time': setup_time + run_time,
                'error': str(e)
            }


def run_incremental_sizes(max_zones=100, zone_increment=10, max_transfers=None, transfer_increment=10, output_csv="incremental_timing_results.csv", show_graph=False):
    """
    Run incremental model sizes and measure setup times.
    
    Args:
        max_zones: Maximum number of zones to test (default: 100)
        zone_increment: Increment for zones (default: 10)
        max_transfers: Maximum number of transfers per zone size (defaults to max_zones, but overrides zone limitation if specified)
        transfer_increment: Increment for transfers (default: 10)
        output_csv: Output CSV filename for results
        show_graph: Show a terminal graph of time vs transfers (default: False)
    """
    from .model_creation import SyntheticModelCreator
    import tempfile
    
    # Track whether max_transfers was explicitly specified
    max_transfers_specified = max_transfers is not None
    
    # Default max_transfers to max_zones if not specified
    if max_transfers is None:
        max_transfers = max_zones
    
    results = []
    
    print(f"Running incremental size tests up to {max_zones} zones and {max_transfers} transfers...")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv) if os.path.dirname(output_csv) else "."
    if output_dir and output_dir != "." and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate through zone sizes
    for zones in range(zone_increment, max_zones + 1, zone_increment):
        print(f"\nTesting with {zones} zones...")
        
        # Determine max transfers for this zone size
        if max_transfers_specified:
            # If max_transfers was explicitly specified, use it regardless of zone count
            max_transfers_for_zones = max_transfers
        else:
            # If max_transfers was not specified, limit to zone count (original behavior)
            max_transfers_for_zones = min(max_transfers, zones)
            
        for transfers in range(transfer_increment, max_transfers_for_zones + 1, transfer_increment):
            print(f"  Creating model with {zones} zones and {transfers} transfers...")
            
            try:
                # Create the model
                creator = SyntheticModelCreator(zones, transfers)
                
                # Ensure models directory exists
                models_dir = "models"
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)
                    print(f"    Created models directory: {os.path.abspath(models_dir)}")
                
                # Create a named JSON file for the current model in the models folder
                model_filename = os.path.join(models_dir, f"current_model_{zones}z_{transfers}t.json")
                model_data = creator.build_model()
                
                print(f"    Creating model file: {model_filename}")
                with open(model_filename, 'w') as f:
                    json.dump(model_data, f, indent=2)
                
                print(f"    Model file created: {os.path.abspath(model_filename)}")
                
                # Run the model and capture timing (without creating CSV output)
                pfr = PywrFileRunner()
                pfr.load_pywr_model_from_file(model_filename)
                timing_info = pfr.run_pywr_model(create_csv=False)
                
                # Store results
                result = {
                    'zones': zones,
                    'transfers': transfers,
                    'setup_time': timing_info.get('setup_time', 0),
                    'run_time': timing_info.get('run_time', 0),
                    'total_time': timing_info.get('total_time', 0),
                    'error': timing_info.get('error', '')
                }
                results.append(result)
                
                print(f"    Setup time: {timing_info.get('setup_time', 0):.3f}s")
                
                # Show updated graph if requested (after each model run)
                if show_graph and len(results) > 0:
                    current_df = pd.DataFrame(results)
                    _display_terminal_graph(current_df)
                
                # Clean up the model file after successful run
                try:
                    os.unlink(model_filename)
                    print(f"    Cleaned up: {model_filename}")
                except OSError as e:
                    print(f"    Warning: Could not delete {model_filename}: {e}")
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                
                # Clean up model file if it exists, even on error
                models_dir = "models"
                model_filename = os.path.join(models_dir, f"current_model_{zones}z_{transfers}t.json")
                if os.path.exists(model_filename):
                    try:
                        os.unlink(model_filename)
                        print(f"    Cleaned up after error: {model_filename}")
                    except OSError:
                        pass
                
                result = {
                    'zones': zones,
                    'transfers': transfers,
                    'setup_time': 0,
                    'run_time': 0,
                    'total_time': 0,
                    'error': str(e)
                }
                results.append(result)
                
                # Show updated graph even after errors if requested
                if show_graph and len(results) > 0:
                    current_df = pd.DataFrame(results)
                    _display_terminal_graph(current_df)
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        output_path = os.path.join(output_dir, output_csv)
        try:
            df.to_csv(output_path, index=False)
            print(f"\nResults saved to {os.path.abspath(output_path)}")

            # Verify file was created
            if os.path.exists(output_path):
                print(f"✓ Output file confirmed: {os.path.abspath(output_path)}")
                file_size = os.path.getsize(output_path)
                print(f"  File size: {file_size} bytes")
            else:
                print(f"✗ Warning: Output file was not created at {output_path}")

        except Exception as e:
            print(f"✗ Error saving CSV: {str(e)}")
            raise
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"Total models tested: {len(results)}")
        if len(results) > 0:
            print(f"Average setup time: {df['setup_time'].mean():.3f}s")
            print(f"Max setup time: {df['setup_time'].max():.3f}s")
            print(f"Min setup time: {df['setup_time'].min():.3f}s")

        return df
    else:
        print("No results to save.")
        return pd.DataFrame()


def _display_terminal_graph(df):
    """Display a 2D line chart of setup time vs transfers in the terminal."""
    print(f"\n{'='*80}")
    print(f"SETUP TIME vs TRANSFERS (2D Line Chart) - {len(df)} models tested")
    print(f"{'='*80}")
    
    if df.empty:
        print("No data to plot.")
        return
    
    # Get unique zone counts for different lines
    zone_counts = sorted(df['zones'].unique())
    
    # Determine overall ranges for consistent scaling
    all_transfers = df['transfers'].tolist()
    all_setup_times = df['setup_time'].tolist()
    
    if not all_transfers or not all_setup_times:
        print("No valid data to plot.")
        return
    
    min_transfers = min(all_transfers)
    max_transfers = max(all_transfers)
    min_time = min(all_setup_times)
    max_time = max(all_setup_times)
    
    # Chart dimensions
    chart_width = 60
    chart_height = 20
    
    # Create the 2D grid
    grid = [[' ' for _ in range(chart_width)] for _ in range(chart_height)]
    
    # Characters for different zone counts
    zone_chars = ['●', '■', '▲', '♦', '★', '◆', '▼', '◄', '►', '♠']
    
    # Plot data for each zone count
    legend_info = []
    
    for i, zones in enumerate(zone_counts):
        zone_data = df[df['zones'] == zones].sort_values('transfers')
        if zone_data.empty:
            continue
        
        char = zone_chars[i % len(zone_chars)]
        legend_info.append((zones, char))
        
        # Plot points for this zone count
        prev_x = prev_y = None
        
        for _, row in zone_data.iterrows():
            transfers = row['transfers']
            setup_time = row['setup_time']
            
            # Scale to grid coordinates
            if max_transfers > min_transfers:
                x = int(((transfers - min_transfers) / (max_transfers - min_transfers)) * (chart_width - 1))
            else:
                x = chart_width // 2
                
            if max_time > min_time:
                y = chart_height - 1 - int(((setup_time - min_time) / (max_time - min_time)) * (chart_height - 1))
            else:
                y = chart_height // 2
            
            # Ensure coordinates are within bounds
            x = max(0, min(chart_width - 1, x))
            y = max(0, min(chart_height - 1, y))
            
            # Plot the point
            grid[y][x] = char
            
            # Draw line from previous point if exists
            if prev_x is not None and prev_y is not None:
                _draw_line(grid, prev_x, prev_y, x, y, char, chart_width, chart_height)
            
            prev_x, prev_y = x, y
    
    # Print the chart
    print(f"\nSetup Time (s)")
    print(f"^")
    
    # Y-axis labels and grid
    for y in range(chart_height):
        # Calculate the time value for this y position
        time_val = max_time - ((y / (chart_height - 1)) * (max_time - min_time)) if max_time > min_time else min_time
        
        # Print y-axis label every few lines
        if y % 4 == 0:
            print(f"{time_val:6.3f} |", end="")
        else:
            print(f"       |", end="")
        
        # Print the row
        print(''.join(grid[y]))
    
    # X-axis
    print(f"       +{'-' * chart_width}")
    print(f"       ", end="")
    
    # X-axis labels
    for i in range(0, chart_width, 10):
        if max_transfers > min_transfers:
            transfer_val = min_transfers + ((i / (chart_width - 1)) * (max_transfers - min_transfers))
        else:
            transfer_val = min_transfers
        print(f"{transfer_val:8.0f}", end="  ")
    
    print(f"\n{' ' * 35}Transfers")
    
    # Legend
    print(f"\nLegend:")
    for zones, char in legend_info:
        print(f"  {char} = {zones} zones")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"Transfer range: {min_transfers} - {max_transfers}")
    print(f"Setup time range: {min_time:.3f}s - {max_time:.3f}s")
    
    # Show correlation for each zone count
    if len(zone_counts) > 1:
        print(f"\nCorrelations by zone count:")
        for zones in zone_counts:
            zone_data = df[df['zones'] == zones]
            if len(zone_data) > 1:
                correlation = zone_data['transfers'].corr(zone_data['setup_time'])
                if not pd.isna(correlation):
                    print(f"  {zones} zones: {correlation:.3f}")
    
    # Overall correlation
    if len(df) > 1:
        overall_correlation = df['transfers'].corr(df['setup_time'])
        if not pd.isna(overall_correlation):
            print(f"\nOverall correlation: {overall_correlation:.3f}")
            if overall_correlation > 0.7:
                print("  → Strong positive correlation")
            elif overall_correlation > 0.3:
                print("  → Moderate positive correlation")
            elif overall_correlation < -0.7:
                print("  → Strong negative correlation")
            elif overall_correlation < -0.3:
                print("  → Moderate negative correlation")
            else:
                print("  → Weak correlation")
    
    print(f"{'='*80}")


def _draw_line(grid, x1, y1, x2, y2, char, width, height):
    """Draw a line between two points on the grid using Bresenham's algorithm."""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    x, y = x1, y1
    
    x_inc = 1 if x1 < x2 else -1
    y_inc = 1 if y1 < y2 else -1
    
    error = dx - dy
    
    while True:
        # Ensure coordinates are within bounds
        if 0 <= x < width and 0 <= y < height:
            if grid[y][x] == ' ':  # Don't overwrite existing points
                grid[y][x] = '·'  # Use a different character for line segments
        
        if x == x2 and y == y2:
            break
            
        error2 = 2 * error
        
        if error2 > -dy:
            error -= dy
            x += x_inc
            
        if error2 < dx:
            error += dx
            y += y_inc

