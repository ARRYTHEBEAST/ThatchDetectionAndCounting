import os
import glob
import shutil
from ultralytics import YOLO
import pandas as pd

# Define paths
base_weights_path = "/content/yolov8/runs/train/yolov8l6-1536-image-size-25-epoch-fold-"
base_image_path = "dataset/fold_{}/images"
save_path = "/content/yolov8l6-1536-image-size-25-epoch-mskf"

# Load the model for each fold and run inference
for i in range(5):
    model = YOLO(f"{base_weights_path}{i}/weights/best.pt")
    model.to('cuda:0')

    results = model.predict(
        source=base_image_path.format(i),
        img_size=1536,
        conf=0.1,
        half=True,
        save_txt=True,
        save_conf=True,
        nosave=True,
        augment=True,
        project="runs/detect",
        name=f"yolov8-custom-different-augs-image-size-1024-{i}_val"
    )

# Collect and move prediction files
preds_txt = glob.glob('runs/detect/yolov8-custom-different-augs-image-size-1024-*_val/labels/*.txt')
os.makedirs(save_path, exist_ok=True)
for file_path in preds_txt:
    shutil.copy(file_path, save_path)
    os.remove(file_path)

print(f"All predictions are saved to {save_path}")

# Combine predictions into a single DataFrame and process them
predictions = []
for file_path in glob.glob(os.path.join(save_path, "*.txt")):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            image_id = os.path.basename(file_path).replace('.txt', '')
            class_id = int(parts[0])
            confidence = float(parts[5])
            predictions.append([f"{image_id}_{class_id}", confidence])

# Create DataFrame
df_predictions = pd.DataFrame(predictions, columns=["image_id", "Target"])

# Sample output
print(df_predictions.head())

# Save the combined predictions
output_file = os.path.join(save_path, 'combined_predictions.csv')
df_predictions.to_csv(output_file, index=False)
print(f"Combined predictions saved to {output_file}")
