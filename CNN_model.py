import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CONV(nn.Module):
    # inherited nn model from torch
    def __init__(self, input_channels, num_classes):
        super().__init__()
        # set the CONV layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # set the pooling to reduce half, select max pooling
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv2D(32)
        x = self.pool(F.relu(self.conv2(x)))  # Conv2D(64)
        x = self.pool(F.relu(self.conv3(x)))  # Conv2D(128)
        x = self.pool(F.relu(self.conv4(x)))  # Conv2D(128)
        x = x.view(x.size(0), -1)             # Flatten
        x = self.dropout(F.relu(self.fc1(x)))              # Dense(512)
        x = self.fc2(x)                       # Dense(class_indices)
        return x          # Softmax output
    
class CONV_ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # use pretrain model
        self.model = models.resnet18(pretrained = True)
        
        # replace final layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
