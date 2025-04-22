# oversample_utils.py

from collections import defaultdict
from random import choice

def oversample_with_augmentation_flags(images, labels, target_count=800):
    class_to_images = defaultdict(list)
    for img, label in zip(images, labels):
        class_to_images[label].append(img)

    new_images, new_labels, augment_flags = [], [], []

    for cls, img_list in class_to_images.items():
        original_count = len(img_list)

        new_images.extend(img_list)
        new_labels.extend([cls] * original_count)
        augment_flags.extend([False] * original_count)

        while len(img_list) < target_count:
            img = choice(img_list)
            new_images.append(img)
            new_labels.append(cls)
            augment_flags.append(True)
            img_list.append(img) 

    return new_images, new_labels, augment_flags
