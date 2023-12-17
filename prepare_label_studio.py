import os
import shutil


def organize_images(main_folder, num_images_per_folder=60):
    # Ensure main_folder exists
    if not os.path.exists(main_folder):
        print(f"Main folder '{main_folder}' does not exist.")
        return

    # List all files in the main folder
    all_files = [f for f in os.listdir(main_folder) if os.path.isfile(os.path.join(main_folder, f))]

    # Filter out only image files if necessary (e.g., jpg, png)
    # all_files = [f for f in all_files if f.lower().endswith(('.jpg', '.png'))]

    # Create subfolders and move files
    for i in range(0, len(all_files), num_images_per_folder):
        subfolder_name = str(i // num_images_per_folder + 1)  # Folder names: 1, 2, 3, ...
        subfolder_path = os.path.join(main_folder, subfolder_name)

        # Create subfolder if it doesn't exist
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Move files to the subfolder
        for file in all_files[i:i + num_images_per_folder]:
            shutil.move(os.path.join(main_folder, file), os.path.join(subfolder_path, file))

# Example usage
organize_images('/home/kasra/kasra_files/data-shenasname/2000_new_data')

# Replace 'path_to_main_folder' with the actual path to your main folder.
# This script will create subfolders named '1', '2', '3', etc., in the main folder,
# and move 60 images to each subfolder.
