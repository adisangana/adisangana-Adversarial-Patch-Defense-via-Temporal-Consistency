import sys
from pathlib import Path
from PIL import Image

def fix_bgr_to_rgb(dataset_dir: str):
    """
    Recursively finds all PNG files in the given directory and swaps 
    the Red and Blue color channels to fix BGR/RGB conversion errors.
    """
    base_path = Path(dataset_dir)
    if not base_path.exists():
        print(f"Error: Path '{dataset_dir}' does not exist.")
        return

    # Find all PNG files recursively in subdirectories
    png_files = list(base_path.rglob("*.png"))
    if not png_files:
        print(f"No PNG files found in {base_path}")
        return

    print(f"Found {len(png_files)} images. Swapping Red and Blue channels...")
    
    for img_path in png_files:
        try:
            with Image.open(img_path) as img:
                # Ensure we are working with an RGB image
                img = img.convert('RGB')
                
                # Split the image into Red, Green, and Blue channels
                r, g, b = img.split()
                
                # Merge them back together, but swap the R and B channels
                fixed_img = Image.merge('RGB', (b, g, r))
                
                # Overwrite the original image
                fixed_img.save(img_path)
                
            print(f"Fixed colors for: {img_path.parent.name}/{img_path.name}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    print("\nColor channel swap complete!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_color_channels.py <path_to_dataset>")
        sys.exit(1)
    
    fix_bgr_to_rgb(sys.argv[1])
