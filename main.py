from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv

from data_processor import load_data, preprocess_for_synthesis
from validator import run_evaluation_report
from approaches.jittering_model import generate_jittered_data

load_dotenv()

def main():
    # Define directory paths
    data_dir = Path("data")
    results_dir = Path("results")

    # File paths
    input_file = data_dir / "data.csv"
    
    # Phase 1: Data loading
    if not input_file.exists():
        print(f"Error: {input_file} not found. Please place the CSV in the 'data' folder.")
        return

    raw_data = load_data(str(input_file))
    df_clean = preprocess_for_synthesis(raw_data)

    # Phase 2: Synthetic Data Generation using Jittering
    print("\nGenerating synthetic data using jittering approach...")
    num_samples = len(raw_data)
    synthetic_df = generate_jittered_data(df_clean, num_rows=num_samples, noise_level=0.01)

    # Phase 3: Quality Assurance
    run_evaluation_report(raw_data, synthetic_df)

    # Phase 4: Export
    os.makedirs(results_dir, exist_ok=True)
    output_path = results_dir / "submission.csv"
    synthetic_df.to_csv(output_path, index=False)

    print(f"\nExecution successful.")
    print(f"Synthetic file saved to: {output_path}")

if __name__ == "__main__":
    main()