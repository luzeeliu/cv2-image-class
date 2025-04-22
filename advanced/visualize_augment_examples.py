import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from random import choice

# ===== 设置图像路径 =====
# 从你的训练集路径中任意挑选一张图
sample_dir = "../Aerial_Landscapes/airport"  # 替换为你实际路径中的任意类
sample_image_path = choice([os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith(".jpg")])

image = cv2.imread(sample_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ===== 定义原始增强 transform =====
basic_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

# ===== 定义强增强 transform（用于 oversample 样本）=====
strong_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
    transforms.ToTensor(),
])

# ===== 可视化函数 =====
def visualize_transformed_versions(image_np, transform, title):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        tensor = transform(image_np)
        img = tensor.permute(1, 2, 0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # 反归一化
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"{title} {i+1}")
    plt.tight_layout()
    plt.show()

# ===== 显示效果 =====
print(f"🔍 Showing examples from: {sample_image_path}")
visualize_transformed_versions(image, basic_transform, "Original Augment")
visualize_transformed_versions(image, strong_transform, "Strong Augment")
