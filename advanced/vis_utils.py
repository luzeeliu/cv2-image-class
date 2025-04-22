import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from grad_cam import apply_gradcam_on_image
import os
import cv2


# ==== Plot Training Curves ====
def plot_training_curves(history, save_path="train_val_accuracy.png"):
    """
    Plot training and validation accuracy and loss curves.

    Args:
        history: Dictionary containing 'train_acc', 'test_acc', 'train_loss', and 'test_loss'.
        save_path: File path to save the plot image.
    """
    epochs = range(1, len(history['train_acc']) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['test_acc'], label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved at: {save_path}")


# ==== Plot Confusion Matrix ====
def plot_confusion_matrix_dl(model, dataloader, classes, device, save_path="confusion_matrix.png"):
    """
    Plot confusion matrix for deep learning model predictions.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader for test data.
        classes: List of class names.
        device: Computation device ('cuda' or 'cpu').
        save_path: File path to save the confusion matrix plot.
    """
    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved at: {save_path}")


# ==== Plot Misclassified Samples ====
def plot_misclassified_samples_dl(model, dataloader, classes, device, save_path="misclassified_examples.png", max_samples=25):
    """
    Plot examples of misclassified samples by the model.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader for test data.
        classes: List of class names.
        device: Computation device ('cuda' or 'cpu').
        save_path: File path to save misclassified samples plot.
        max_samples: Maximum number of misclassified samples to plot.
    """
    misclassified = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(inputs.size(0)):
                if preds[i] != labels[i]:
                    misclassified.append((inputs[i].cpu(), labels[i].item(), preds[i].item()))

    if not misclassified:
        print("No misclassified samples found.")
        return

    n_show = min(len(misclassified), max_samples)
    cols = 5
    rows = (n_show + cols - 1) // cols

    plt.figure(figsize=(15, 3 * rows))

    for idx in range(n_show):
        img_tensor, true_label, pred_label = misclassified[idx]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img_np = np.clip(img_np, 0, 1)

        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img_np)
        plt.axis('off')
        plt.title(f"True: {classes[true_label]}\nPred: {classes[pred_label]}", fontsize=9, color='red')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f" Misclassified samples saved at: {save_path}")

def generate_gradcam_for_misclassified(model, dataloader, classes, device, target_layer, transform, save_dir="gradcam_errors", max_samples=10):
    """
    Generate Grad-CAM visualizations for misclassified samples in the dataset.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader providing input images and labels.
        classes: List of class names, indexed by label.
        device: Computation device (e.g., 'cuda' or 'cpu').
        target_layer: Layer in the model to apply Grad-CAM on.
        transform: Preprocessing function that converts raw image to model input.
        save_dir: Directory to save Grad-CAM images.
        max_samples: Maximum number of misclassified samples to visualize.
    """
    os.makedirs(save_dir, exist_ok=True)  # Create output directory if it doesn't exist
    model.eval()  # Set model to evaluation mode
    count = 0     # Counter for saved misclassified samples

    with torch.no_grad():  # Disable gradient computation for faster inference
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get predicted class indices

            for i in range(inputs.size(0)):
                if preds[i] != labels[i]:  # Check if the prediction is incorrect
                    # Convert tensor image to numpy array for visualization
                    raw_tensor = inputs[i].cpu()
                    image_np = raw_tensor.permute(1, 2, 0).numpy()
                    
                    # De-normalize image (reverse ImageNet normalization)
                    image_np = image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
                    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
                    
                    # Convert RGB to BGR for OpenCV compatibility
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                    # Build the file path with true and predicted class names
                    save_path = os.path.join(
                        save_dir,
                        f"error_{count}_T_{classes[labels[i]]}_P_{classes[preds[i]]}.png"
                    )

                    # Apply Grad-CAM to the misclassified image
                    apply_gradcam_on_image(
                        model=model,
                        image_bgr=image_bgr,
                        label=labels[i].item(),
                        device=device,
                        target_layer=target_layer,
                        preprocess_fn=transform,
                        save_path=save_path
                    )

                    count += 1
                    if count >= max_samples:
                        print(f"Saved the first {max_samples} misclassified samples with Grad-CAM")
                        return
