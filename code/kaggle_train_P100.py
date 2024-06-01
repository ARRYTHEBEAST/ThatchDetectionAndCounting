# Disable WandB
#os.environ['WANDB_MODE'] = 'disabled'

from ultralytics import YOLO

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