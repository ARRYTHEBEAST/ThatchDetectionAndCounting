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



#### HyperParameter Tuning




import os
import glob
import shutil
import optuna
import pandas as pd
import numpy as np
from statistics import mean
from ensemble_boxes import *
from pybboxes import BoundingBox
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import yaml

# Update these paths to match your data
train_df = pd.read_csv('Train.csv')
train_labels_df = pd.read_csv('train_modified.csv')
pred_labels_path = '/path/to/yolov8_predictions'
classifier_pred = pd.read_csv('classifier_predictions.csv')

# Create dictionaries for quick look-up
id_label_dict = dict(zip(train_labels_df['image_id'].values, train_labels_df['number_of_objects'].values))
classifier_pred_dict = dict(zip(classifier_pred['id'].values, classifier_pred['pred'].values))

def mae(y_true, y_pred):
    return np.abs(y_true - y_pred)

def return_error(id, pred_label_dict):
    id = id.split('.')[0]
    true_label = id_label_dict[f'{id}.jpg']
    pred_label = pred_label_dict[f'{id}.jpg']
    error = float(mae(np.array(true_label), np.array(pred_label)))
    return error

def make_labels(id, params):
    id = id.split('.')[0]
    class0 = 0
    class1 = 0
    class2 = 0

    classifier_pred = classifier_pred_dict[id] * 1.0

    if os.path.exists(f'{pred_labels_path}/{id}.txt') and classifier_pred > params['classifier_thresh']:
        with open(f'{pred_labels_path}/{id}.txt') as f:
            preds_per_line = f.readlines()
            bboxes = []
            scores = []
            label = []

            for i in preds_per_line:
                i = i.split(' ')
                bbox = [float(i[1]), float(i[2]), float(i[3]), float(i[4])]
                bbox = BoundingBox.from_yolo(*bbox, image_size=(1536, 1536))
                bbox = bbox.to_albumentations().values

                bboxes.append(list(bbox))
                scores.append(float(i[5]))
                label.append(int(i[0]))

            bboxes, scores, label = soft_nms([bboxes], [scores], [label], iou_thr=params['iou_thr'],
                                             sigma=params['sigma'], thresh=params['thresh'], method=params['method'])

            for i in range(len(label)):
                if label[i] == 0:
                    class0 += 1
                elif label[i] == 1:
                    class1 += 1
                elif label[i] == 2:
                    class2 += 1

    return class0, class1, class2, f"{id}_class0.jpg", f"{id}_class1.jpg", f"{id}_class2.jpg"

class error_func:
    def __init__(self, pred_label_dict):
        self.pred_label_dict = pred_label_dict

    def return_error(self, id):
        id = id.split('.')[0]
        class0_label = id_label_dict[f'{id}_class0.jpg']
        class1_label = id_label_dict[f'{id}_class1.jpg']
        class2_label = id_label_dict[f'{id}_class2.jpg']
        class0_pred = self.pred_label_dict[f'{id}_class0.jpg']
        class1_pred = self.pred_label_dict[f'{id}_class1.jpg']
        class2_pred = self.pred_label_dict[f'{id}_class2.jpg']
        error = float(
            mae(np.array(class0_label), np.array(class0_pred)) +
            mae(np.array(class1_label), np.array(class1_pred)) +
            mae(np.array(class2_label), np.array(class2_pred))
        )
        return error

def objective(trial):
    params = {
        'iou_thr': trial.suggest_float('iou_thr', 0.1, 0.7),
        'sigma': trial.suggest_float('sigma', 0.3, 1.0),
        'thresh': trial.suggest_float('thresh', 0.2, 0.6),
        'method': trial.suggest_categorical('method', ['nms', 'linear', 'gaussian']),
        'classifier_thresh': trial.suggest_float('classifier_thresh', 0.1, 0.7),
    }

    pred = Parallel(n_jobs=12)(delayed(make_labels)(id, params) for id in tqdm(train_df['image_id'].values))
    ids = []
    labels = []
    for i in pred:
        ids.append(i[3])
        ids.append(i[4])
        ids.append(i[5])
        labels.append(i[0])
        labels.append(i[1])
        labels.append(i[2])
    oof = pd.DataFrame({'image_id': ids, 'label': labels}, index=None)

    pred_label_dict = dict(zip(oof['image_id'].values, oof['label'].values))
    error_fn = error_func(pred_label_dict)

    error = list(map(error_fn.return_error, train_df['image_id'].values))

    return mean(error)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=2000)
best_param_save = study.best_params
best_param_save.update({'best_score': study.best_value})
best_param_save.update({'best_trial': study.best_trial.number})
best_param_save.update({'path': pred_labels_path})

with open(f'{pred_labels_path.split("/")[-1]}.yaml', 'w') as f:
    yaml.dump(best_param_save, f)
