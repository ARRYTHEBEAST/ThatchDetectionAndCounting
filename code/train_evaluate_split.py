import pandas as pd
import os
import shutil

# Read Train.csv to get image_ids
Train = pd.read_csv('Train.csv')
array = Train['image_id']

# Shuffle the image_ids
shuffled_series = array.sample(frac=1)

# Slice the shuffled image_ids
sliced_series = shuffled_series.iloc[:5231]

# Define the paths
images_folder = r'E:\ML\ThatchHouseDetection\pythonProject1\code\data\images\images'  # Path to the folder containing images
validation_images_folder = r'E:\ML\ThatchHouseDetection\pythonProject1\code\data\images\val'  # Path to the folder where validation images will be moved

# Get the filenames (without extension) in the sliced_series
sliced_filenames = set(sliced_series.str.split('.').str[0])

# Iterate over the files in the images folder
for filename in os.listdir(images_folder):
    # Get the filename without extension
    filename_without_extension = os.path.splitext(filename)[0]

    # Check if the filename (without extension) is in the sliced_filenames
    if filename_without_extension in sliced_filenames:
        # Move the image file to the validation images folder
        src = os.path.join(images_folder, filename)
        dst = os.path.join(validation_images_folder, filename)
        shutil.move(src, dst)
        print(f"Moved image {filename} to validation_images folder.")