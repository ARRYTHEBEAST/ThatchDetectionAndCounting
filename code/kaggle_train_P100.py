# Disable WandB
#os.environ['WANDB_MODE'] = 'disabled'

from ultralytics import YOLO
import os
import yaml

# Path to the dataset directory
dataset_dir = "/path/to/your/dataset/"

# List of fold directories
fold_dirs = ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]

# Path to the images directory
images_dir = os.path.join(dataset_dir, "data")

# Names of the classes
classes = ["1", "2", "3"]

# Loop through each fold directory
for i, fold_dir in enumerate(fold_dirs):
    # Create a dictionary for the YAML content
    yaml_content = {
        "names": classes,
        "train": [os.path.join(images_dir, fold, "images") for fold in fold_dirs if fold != fold_dir],
        "val": os.path.join(images_dir, fold_dir, "images")
    }

    # Path to save the YAML file
    yaml_file = f"fold_{i}_config.yaml"

    # Write the YAML content to the file
    with open(yaml_file, 'w') as file:
        yaml.dump(yaml_content, file)

    print(f"Created YAML file for fold {i} at {yaml_file}")

# Load the YOLOv8 model
model = YOLO('yolov8l.pt')  # Specify the YOLOv8 large model

# Train the model using multi-GPU
for i in range(5):  # Assuming fold indices from 0 to 4
    fold = f'data/fold_{i}.yaml'  # Define your data fold
    model.train(
        data=fold,
        epochs=10,  # Set to 10 epochs as per your requirement
        #batch=16,  # Explicit batch size, avoiding auto batch for multi-GPU
        device='0',  # Specify multiple GPUs, e.g., '0,1' for two GPUs
        name=f'yolov8l-1536-10-epoch-fold-{i}',
        seed=42,
        #project=None  # Disable WandB project
    )