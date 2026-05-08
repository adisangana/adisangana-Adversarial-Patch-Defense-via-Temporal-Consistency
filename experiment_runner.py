#!/usr/bin/env python3
import os
import csv
import shutil
import sys
from pathlib import Path
from pipeline import parse_args, main as pipeline_main

def run_experiment():

    base_dataset_dirs = [
        r"path_to_dataset_1",
        r"path_to_dataset_2",
        r"path_to_dataset_3",
        r"path_to_dataset_N",
    ]  # You can add more base dataset paths here
    
    window_sizes = [3, 5, 10]
    lighting_conditions = list(range(0, 260, 10))

    base_out_dir = Path("experiment_results")
    base_out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file_path = base_out_dir / "experiment_summary.csv"
    
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "Base Dataset", "Lighting Condition", "Window Size", 
            "Raw Accuracy", "Defended Accuracy", 
            "Attack Detection Rate", "False Positive Rate", 
            "Summary Plot Path"
        ])

        for base_dataset in base_dataset_dirs:
            base_dataset_path = Path(base_dataset)

            for lighting in lighting_conditions:
                # Account for subfolders labelled by lighting condition (e.g., .../clean_30/130)
                dataset_path = base_dataset_path / str(lighting)
                
                if not dataset_path.exists():
                    print(f"Warning: Dataset path {dataset_path} not found. Skipping.")
                    continue

                for w_size in window_sizes:
                    run_name = f"dataset_{base_dataset_path.name}_light{lighting}_w{w_size}"
                    run_out_dir = base_out_dir / run_name
                    run_out_dir.mkdir(parents=True, exist_ok=True)
                    
                    json_report_path = run_out_dir / "report.json"

                    # Override sys.argv to simulate command line arguments for pipeline.py
                    # Note: we pass the specific lighting subfolder as the dataset, 
                    # and the brightness-thresh is handled by pipeline.py defaults or can be explicitly passed
                    sys.argv = [
                        "pipeline.py",
                        "--dataset", str(dataset_path),
                        "--window-size", str(w_size),
                        # Depending on how pipeline.py handles it, you might still want to pass brightness-thresh
                        # If the lighting condition folder implies the thresh:
                        "--brightness-thresh", str(lighting), 
                        "--out-dir", str(run_out_dir),
                        "--no-plots", # Don't show interactive plots, just save them
                        "--report-json", str(json_report_path)
                    ]
                    
                    print(f"\n--- Running Experiment: {run_name} ---")
                    try:
                        # Call the main function of pipeline.py
                        pipeline_main()
                        
                        # Read the JSON report to extract metrics for the CSV
                        if json_report_path.exists():
                            import json
                            with open(json_report_path, 'r') as f:
                                metrics = json.load(f)
                            
                            summary_plot_path = run_out_dir / "summary.png"
                            
                            # Log metrics to the CSV file
                            writer.writerow([
                                base_dataset_path.name,
                                lighting,
                                w_size,
                                metrics.get("raw_accuracy", "N/A"),
                                metrics.get("defended_accuracy", "N/A"),
                                metrics.get("attack_detection_rate", "N/A"),
                                metrics.get("false_positive_rate", "N/A"),
                                str(summary_plot_path) if summary_plot_path.exists() else "Missing"
                            ])
                            
                            # Rename the summary plot to group by dataset, lighting, and window size
                            if summary_plot_path.exists():
                                organized_plot_name = f"summary_{base_dataset_path.name}_light{lighting}_w{w_size}.png"
                                organized_plot_path = base_out_dir / organized_plot_name
                                shutil.copy2(summary_plot_path, organized_plot_path)

                        else:
                            print(f"Error: JSON report not found for {run_name}")
                            writer.writerow([base_dataset_path.name, lighting, w_size, "ERROR", "ERROR", "ERROR", "ERROR", "ERROR"])

                    except Exception as e:
                        print(f"Exception occurred during run {run_name}: {e}")
                        writer.writerow([base_dataset_path.name, lighting, w_size, "FAILED", "FAILED", "FAILED", "FAILED", "FAILED"])

    print(f"\nAll experiments complete. Summary saved to {csv_file_path}")

if __name__ == "__main__":
    run_experiment()