import os
import cv2
import numpy as np
from glob import glob

from torch.utils.data import Dataset
from torchvision import transforms

# Set random seed (used for shuffling paths)
RANDOM_STATE = 42


def load_and_split_dataset(dataset_path, test_size=0.2, sample_ratio=1.0):
    """
    Load dataset from directory and split it into training and testing sets.

    Parameters:
        dataset_path: Root path of the dataset, each subdirectory represents a class
        test_size: Fraction of data reserved for testing (between 0.0 and 1.0)
        sample_ratio: Either a float (uniform sampling ratio) or a list (individual sampling ratio per class)

    Returns:
        (train_images, train_labels), (test_images, test_labels), classes
    """
    train_images, train_labels = [], []
    test_images, test_labels = [], []

    classes = sorted(os.listdir(dataset_path))  # Sort class names alphabetically

    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_path, class_name)
        image_paths = glob(os.path.join(class_dir, '*.jpg'))

        if len(image_paths) == 0:
            continue  # Skip empty classes

        np.random.seed(RANDOM_STATE)
        np.random.shuffle(image_paths)

        #  Supports uniform or per-class sampling
        if isinstance(sample_ratio, float):
            sampled = image_paths[:max(int(len(image_paths) * sample_ratio), 1)]
        elif isinstance(sample_ratio, list) and len(sample_ratio) == len(classes):
            ratio = sample_ratio[idx]
            sampled = image_paths[:max(int(len(image_paths) * ratio), 1)]
        else:
            raise ValueError("sample_ratio must be a float or a list matching the number of classes")

        # Split data into train/test
        split_idx = int(len(sampled) * (1 - test_size))
        train_paths = sampled[:split_idx]
        test_paths = sampled[split_idx:]

        train_images += train_paths
        train_labels += [idx] * len(train_paths)
        test_images += test_paths
        test_labels += [idx] * len(test_paths)

    return (train_images, train_labels), (test_images, test_labels), classes


#  Custom dataset class (used with DataLoader)
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, augment_flags=None, augment_transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.augment_flags = augment_flags if augment_flags is not None else [False] * len(images)
        self.augment_transform = augment_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augment_flags[idx] and self.augment_transform:
            image = self.augment_transform(image)
        elif self.transform:
            image = self.transform(image)

        return image, label



# Data augmentation pipelines for training and testing
def get_transforms(input_size=224):
    """
    Get data augmentation transforms for training and testing datasets (normalized based on ImageNet).

    Returns:
        train_transform: Transform with augmentations
        test_transform: Simple scaling and center cropping without augmentations
    """
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform