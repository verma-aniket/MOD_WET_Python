# import base libraries
import sys
from pathlib import Path

# link core repo folder
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# load model class
from src.model_driver import MODWETModel

def test_simulation_run():

    # 1. Define input and output file paths
    basin_data_path = REPO_ROOT / "data/preprocessed_inputs/high_elev_basin_static_data.nc"
    met_forcing_path = REPO_ROOT / "data/preprocessed_inputs/high_elev_met_forcing.nc"
    output_filepath = REPO_ROOT / "data/output/high_elev_simulation_results.nc"

    # 2. Initialize the model
    print("Initializing MOD-WET model...")
    model = MODWETModel(basin_data_path=basin_data_path, met_forcing_path=met_forcing_path)

    # 3. Execute full simulation and export NetCDF results
    print("Starting simulation run...")
    model.run_simulation(output_filepath=output_filepath)

    # 4. Verify results export
    assert output_filepath.exists(), f"Output file was not created at {output_filepath}"
    assert output_filepath.stat().st_size > 0, "Output file was created but is empty."

    print(f"\nSimulation run and export completed successfully!")

if __name__ == "__main__":
    test_simulation_run()