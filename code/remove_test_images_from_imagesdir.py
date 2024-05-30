import os
import shutil
import pandas as pd

# Define the paths
csv_path = 'Test.csv'
images_dir = r'E:\ML\ThatchHouseDetection\pythonProject1\code\data\images\images'  # Folder containing the images
test_images_dir = r'E:\ML\ThatchHouseDetection\pythonProject1\code\data\images\test_images'  # Folder to move the images to

# Create the test_images directory if it does not exist
if not os.path.exists(test_images_dir):
    os.makedirs(test_images_dir)

# Read the CSV file
image_ids = pd.read_csv(csv_path)['image_id']

# Convert image IDs to a set for faster lookup
image_ids_set = set(image_ids)

# Get list of files in the images directory
image_files = os.listdir(images_dir)

# Process each file in the images directory
for image_file in image_files:
    # Extract the image_id from the file name
    image_id = os.path.splitext(image_file)[0]  # Remove the file extension

    # Check if this image_id is in the set of IDs to be moved
    if image_id in image_ids_set:
        # Construct full file path
        src_path = os.path.join(images_dir, image_file)
        dest_path = os.path.join(test_images_dir, image_file)

        # Move the file
        try:
            shutil.move(src_path, dest_path)
            print(f"Moved: {image_file} to {test_images_dir}")
        except Exception as e:
            print(f"Error moving {image_file}: {e}")

print("Image moving process completed.")