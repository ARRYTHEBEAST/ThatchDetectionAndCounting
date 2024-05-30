import pandas as pd
import os
from PIL import Image

# Define the paths
csv_path = 'Test.csv'
images_dir = r'E:\ML\ThatchHouseDetection\pythonProject1\code\data\Images\images'  # Absolute path to images
labels_dir = r'E:\ML\ThatchHouseDetection\pythonProject1\code\data\labels'  # Absolute path to labels

# Create the labels directory if it does not exist
if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)

# Read the CSV file
annotations = pd.read_csv(csv_path)

# Print first few rows to verify column names and data
print("CSV file columns:", annotations.columns)
print("First few rows of CSV data:\n", annotations.head())

# Function to convert bounding box to YOLO format
def convert_bbox_to_yolo(bbox, img_width, img_height):
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width /= img_width
    height /= img_height
    return x_center, y_center, width, height

# Process each image
for image_id in annotations['image_id'].unique():
    image_name = f"{image_id}.tif"  # Assuming image files are named as 'image_id.tif'
    image_path = os.path.join(images_dir, image_name)

    # Debug: Print the image path
    print(f"Processing image: {image_path}")

    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        continue

    # Get all bounding boxes for the current image
    image_annotations = annotations[annotations['image_id'] == image_id]

    # Create a corresponding text file
    label_path = os.path.join(labels_dir, image_name.replace('.tif', '.txt'))
    with open(label_path, 'w') as f:
        if image_annotations.empty or image_annotations['bbox'].isnull().all():
            print(f"No valid annotations found in image {image_id}, creating empty annotation file")
            continue  # Just create the empty file and move on
        for _, row in image_annotations.iterrows():
            try:
                # Check if bbox is already a list
                if isinstance(row['bbox'], str):
                    bbox = eval(row['bbox'])  # Convert string representation of list to actual list
                else:
                    bbox = row['bbox']  # Directly use if it's already a list
                # Check if category_id is NaN
                if pd.isnull(row['category_id']):
                    print(f"Skipping annotation for image {image_id} due to missing category_id")
                    continue
                class_id = int(row['category_id']) - 1  # Class numbers should be zero-indexed
                # Convert bounding box to normalized xywh format
                x_center, y_center, width, height = convert_bbox_to_yolo(bbox, img_width, img_height)
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            except Exception as e:
                print(f"Error processing annotation for image {image_id}: {e}")

print("Conversion completed.")