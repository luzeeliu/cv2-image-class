# ==== Imports ====
import os
import copy
import random
import torch
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.utils.class_weight import compute_class_weight

from preprocess import load_and_split_dataset, CustomDataset, get_transforms
from oversample_utils import oversample_with_augmentation_flags
from longtail_config import generate_longtail_ratios
from vis_utils import (
    plot_training_curves,
    plot_confusion_matrix_dl,
    plot_misclassified_samples_dl,
    generate_gradcam_for_misclassified
)
from grad_cam import apply_gradcam_on_image
from train import train_model, evaluate_on_test_set
from torch.optim.lr_scheduler import ReduceLROnPlateau

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
# ==== CLI Arguments ====
parser = argparse.ArgumentParser(description="Train ResNet without early stopping")
args = parser.parse_args()

# ==== Configuration Parameters ====
input_size = 224

config = {
    "model_name": "resnet",
    "num_epochs": 30,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "test_size": 0.2,
    "sample_ratio": generate_longtail_ratios(15),
    "use_pretrained": True
}

# ==== Paths and Environment Setup ====
dataset_path = "../Aerial_Landscapes"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==== Load Dataset ====
(train_images, train_labels), (test_images, test_labels), classes = load_and_split_dataset(
    dataset_path,
    test_size=config["test_size"],
    sample_ratio=config["sample_ratio"]
)

train_images, train_labels, augment_flags = oversample_with_augmentation_flags(train_images, train_labels, target_count=800)

# ==== Transforms ====
train_transform, test_transform = get_transforms(input_size)
strong_augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==== Dataset and Dataloader ====
train_dataset = CustomDataset(train_images, train_labels, transform=train_transform, augment_flags=augment_flags, augment_transform=strong_augment)
test_dataset = CustomDataset(test_images, test_labels, transform=test_transform)

train_loader = DataLoader(
    train_dataset, batch_size=config["batch_size"],
    shuffle=True, num_workers=0, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=config["batch_size"],
    shuffle=False, num_workers=0, pin_memory=True
)

# ==== Model and Loss ====
model = models.resnet18(pretrained=config["use_pretrained"])
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(model.fc.in_features, len(classes))
)
model = model.to(device)

# ==== Compute Class Weights ====
y_train = np.array(train_labels)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

plt.figure(figsize=(12, 5))
plt.bar([f"Class {i}" for i in range(len(class_weights))], class_weights, color='skyblue')
plt.ylabel("Class Weight")
plt.title("Per-Class Weight Used in Weighted Loss")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/class_weight_distribution.png")
plt.close()

criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
    verbose=True
)

# ==== Train the Model ====
print("\n🚀 Starting training")
model, history = train_model(
    model, device, train_loader, test_loader,
    criterion, optimizer,
    num_epochs=config["num_epochs"],
    scheduler=scheduler
)

# ==== Evaluate and Visualize ====
evaluate_on_test_set(model, test_loader, classes, device, save_path="outputs/report_oversample.txt")
plot_training_curves(history, save_path="outputs/train_val_accuracy.png")
plot_confusion_matrix_dl(model, test_loader, classes, device, save_path="outputs/confusion_matrix.png")
plot_misclassified_samples_dl(model, test_loader, classes, device, save_path="outputs/misclassified_examples.png")
generate_gradcam_for_misclassified(model, test_loader, classes, device, target_layer=model.layer4, transform=test_transform)