import os
import sys
from pathlib import Path

def process_single_brightness_folder(brightness_path: Path):
    """
    Processes a single folder containing images for one brightness level.
    The folder's name is used as the brightness value.
    """
    try:
        brightness = brightness_path.name
        int(brightness) # Validate that the folder name is a number
    except (ValueError):
        print(f"Info: Skipping non-numerical subfolder '{brightness_path.name}'.")
        return

    print(f"\n--- Processing brightness level: {brightness} ---")

    # Collect all .png files and sort them to establish a reliable temporal order
    png_files = sorted([f for f in brightness_path.glob('*.png')])
    if not png_files:
        print(f"No .png files found in '{brightness_path}'.")
        return

    degree_counter = 0
    distance_counter = 0

    for i, old_png_path in enumerate(png_files):
        # Every 30 frames, increment the degree and reset the distance
        if i > 0 and i % 30 == 0:
            degree_counter += 1
            distance_counter = 0

        # Format the new filename
        new_stem = f"brightness{brightness}_degree{degree_counter:02d}_distance{distance_counter:02d}"
        new_png_path = brightness_path / f"{new_stem}.png"
        new_json_path = brightness_path / f"{new_stem}.json"

        # Find the corresponding old .json file
        old_json_path = old_png_path.with_suffix('.json')

        # Skip files that already match the new format to make the script idempotent
        if old_png_path.name == new_png_path.name:
            continue

        print(f"  Renaming '{old_png_path.name}' -> '{new_png_path.name}'")
        
        # Perform the renaming
        try:
            # Use rename, which is atomic on most systems
            old_png_path.rename(new_png_path)
            if old_json_path.exists():
                old_json_path.rename(new_json_path)
            else:
                print(f"  Warning: No corresponding .json file found for '{old_png_path.name}'")

        except OSError as e:
            print(f"  Error renaming files: {e}")
            # Attempt to revert if one file was renamed and the other failed
            if new_png_path.exists() and not old_png_path.exists():
                new_png_path.rename(old_png_path)
            return # Stop processing this folder on error

        distance_counter += 1

    print(f"  Processed {len(png_files)} image files.")

def main():
    """
    Main function to orchestrate the renaming process.
    Takes a parent directory containing subfolders named by brightness level.
    """
    if len(sys.argv) != 2:
        print("Usage: python rename_dataset_files.py <path_to_parent_dataset_folder>")
        print("Example: python rename_dataset_files.py 'path_to_dataset'")
        sys.exit(1)
    
    parent_directory = Path(sys.argv[1])

    if not parent_directory.is_dir():
        print(f"Error: Provided path '{parent_directory}' is not a valid directory.")
        sys.exit(1)

    print(f"Scanning parent directory: '{parent_directory}'")

    # Iterate through all immediate subdirectories
    for subfolder in parent_directory.iterdir():
        if subfolder.is_dir():
            process_single_brightness_folder(subfolder)

    print("\n--- All subfolders processed. ---")

if __name__ == "__main__":
    main()
