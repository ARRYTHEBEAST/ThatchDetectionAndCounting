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
labels_folder = r'E:\ML\ThatchHouseDetection\pythonProject1\code\data\labels\labels'  # Path to the folder containing labels
val_labels_folder = r'E:\ML\ThatchHouseDetection\pythonProject1\code\data\labels\val_labels'  # Path to the folder where validation labels will be moved

# Get the filenames (without extension) in the sliced_series
sliced_filenames = set(sliced_series.str.split('.').str[0])

# Iterate over the files in the labels folder
for filename in os.listdir(labels_folder):
    # Get the filename without extension
    filename_without_extension = os.path.splitext(filename)[0]

    # Check if the filename (without extension) is in the sliced_filenames
    if filename_without_extension in sliced_filenames:
        # Move the label file to the validation labels folder
        src = os.path.join(labels_folder, filename)
        dst = os.path.join(val_labels_folder, filename)
        shutil.move(src, dst)
        print(f"Moved label {filename} to val_labels folder.")