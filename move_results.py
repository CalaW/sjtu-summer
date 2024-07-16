import os
import shutil

# Define the root directory where the folders are located
root_dir = "vis"

# Iterate through each directory in the root directory
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Iterate through each file in the directory
        for file_name in os.listdir(folder_path):
            # Check if the file is a jpg or json file
            if file_name.endswith(".jpg") or file_name.endswith(".json"):
                # Construct the new file name using the parent folder's name
                new_file_name = folder_name + file_name[-4:]
                # Construct the source file path
                src_file_path = os.path.join(folder_path, file_name)
                # Construct the destination file path
                dest_file_path = os.path.join(root_dir, new_file_name)
                # Move and rename the file
                shutil.move(src_file_path, dest_file_path)
