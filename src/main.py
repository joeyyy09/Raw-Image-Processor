from processor.raw_image_processor import RAWImageProcessor
import sys
from pathlib import Path

def main():
    processor = RAWImageProcessor()
    raw_path = "raw.ARW"
    
    if not Path(raw_path).exists():
        print(f"Error: Could not find {raw_path}")
        print("Make sure the RAW file is in the same directory as this script")
        return
    
    processor.process_raw_image(raw_path)
    sys.exit(0)

if __name__ == "__main__":
    main()