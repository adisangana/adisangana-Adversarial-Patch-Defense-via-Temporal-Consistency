import sys
import subprocess
from pathlib import Path

def run_yolo_on_datasets():
    # Define the base datasets you want to process
    base_dataset_dirs = [
        r"path_to_dataset"
    ]
    
    for base_dir in base_dataset_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"Warning: Base dataset path '{base_path}' not found. Skipping.")
            continue
            
        print(f"\n========================================================")
        print(f"Processing base directory: {base_path.name}")
        print(f"========================================================")

        # Iterate through the numerical lighting subfolders (0, 10, 20, etc.)
        for subfolder in sorted(base_path.iterdir()):
            if subfolder.is_dir() and subfolder.name.isdigit():
                print(f"\n--- Running YOLOv5 on {base_path.name}/{subfolder.name} ---")
                
                # Construct the command to run detect.py
                # This will find all PNGs in the subfolder and overwrite their JSONs
                cmd = [
                    sys.executable, 
                    "detect.py",
                    "--dataset", str(subfolder),
                    "--overwrite"
                ]
                
                try:
                    # Execute the command
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error running detect.py on {subfolder}: {e}")

    print("\nAll YOLO detection runs are complete!")

if __name__ == "__main__":
    run_yolo_on_datasets()
