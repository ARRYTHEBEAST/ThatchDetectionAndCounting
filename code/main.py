import os
import pandas as pd
from tqdm.auto import tqdm
from ultralytics import YOLO

# Define paths
test_image_dir = r'E:\ML\ThatchHouseDetection\pythonProject1\code\test\test_images'
weights_path = r"C:\Users\arjun\Downloads\best_4.pt"

# Load the model
model = YOLO(weights_path)
model.to('cuda:0')

# Create a DataFrame to store results
image_files = os.listdir(test_image_dir)

# Initialize a list to store the output data
output_data = []

# Perform inference and collect counts
for img_id in tqdm(image_files):
    img_path = os.path.join(test_image_dir, img_id)
    results = model.predict(source=img_path, conf=0.1, augment=True, half=True, device='cuda:0')

    # Initialize class counts
    class_counts = [0, 0, 0]

    # Update class counts based on detected classes
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id in [0, 1, 2]:
                class_counts[class_id] += 1

    # Create output rows for each class
    image_id_base = os.path.splitext(img_id)[0]
    for class_id in range(3):
        output_data.append({
            'image_id': f"{image_id_base}_{class_id + 1}",
            'Target': class_counts[class_id]
        })

# Convert to DataFrame
output_df = pd.DataFrame(output_data)

# Save to CSV
output_df.to_csv('Submission_4.csv', index=False)
print("Submission file saved as 'Submission_4.csv'")

# Print the formatted DataFrame
print(output_df)
