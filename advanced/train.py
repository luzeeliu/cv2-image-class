# ==== Imports ====
import os
import copy
import torch
import numpy as np
from tqdm import tqdm

# ==== Train Function ====
def train_model(model, device, train_loader, test_loader,
                criterion, optimizer,
                num_epochs=25,
                scheduler=None,
                checkpoint_path='checkpoint.pth'):
    """
    Train a PyTorch model without early stopping.

    Args:
        model: PyTorch model
        device: torch.device
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
        criterion: loss function
        optimizer: optimizer
        num_epochs: total epochs
        checkpoint_path: where to save best model

    Returns:
        best_model: model with best val accuracy
        history: training history
    """
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    log_file = open("outputs/training_log.txt", "w")

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # Validation phase
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
                total += labels.size(0)

        epoch_test_loss = running_loss / total
        epoch_test_acc = correct / total

        # Save metrics
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['test_loss'].append(epoch_test_loss)
        history['test_acc'].append(epoch_test_acc)

        log_line = (f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
                    f"Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.4f}")
        print(log_line)
        log_file.write(log_line + "\n")

        if scheduler is not None:
            scheduler.step(epoch_test_loss)
            
        # Save best model weights
        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    log_file.close()
    model.load_state_dict(best_model_wts)
    return model, history

# ==== Evaluation Function ====
from sklearn.metrics import classification_report

def evaluate_on_test_set(model, dataloader, classes, device, save_path="report.txt"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    report_dict = classification_report(all_labels, all_preds, target_names=classes, digits=4, output_dict=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(report)
    print(f"✅ Classification report saved to {save_path}")

    return report_dict