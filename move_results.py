import shutil
from pathlib import Path

# Define the root directory where the folders are located
root_dir = Path("vis")

for subj_dir in root_dir.glob("*/"):
    # Iterate through each directory in the root directory
    for folder_path in subj_dir.glob("*d/*/"):
        print(folder_path)
        # Iterate through each file in the directory
        for file_path in folder_path.glob("*"):
            print(file_path)
            # Check if the file is a jpg or json file
            if file_path.suffix in (".jpg", ".json"):
                # Construct the source file path
                src_file_path = file_path
                # Construct the destination file path
                dest_file_path = folder_path.with_suffix(file_path.suffix)
                # Move and rename the file
                shutil.move(src_file_path, dest_file_path)
                print(f"Moved {src_file_path} to {dest_file_path}")
        shutil.rmtree(folder_path)
        print(f"Removed {folder_path}")
