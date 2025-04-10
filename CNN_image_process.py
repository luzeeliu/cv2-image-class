import os
import cv2
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

# first we need get the class name from dataset
# use os to manipulate it 
def get_classes(data_dir):
    # the data_dir is train/ 
    ans = []
    for d in os.listdir(data_dir):
        # d is the class label, use join to entry the folder
        if os.path.isdir(os.path.join(data_dir, d)):
            ans.append(d)
    return sorted(ans)

# DATA AUGMENTATION      
def transform(input_size = 224):
    train_transform = transforms.Compose([
        #transforms.ToPILImage(),    # convert input to PIL image
        transforms.RandomResizedCrop(input_size),    # stochastic crop image
        transforms.RandomHorizontalFlip(),  # random horizontal flip the image
        transforms.RandomRotation(15),  # random rotation 15 degree
        transforms.ToTensor(),  # put it to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
       # transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform

class ImageDataset(Dataset):
    def __init__(self, image_path, class_name, transform = None):
        # initial parameter
        # in imagedataset we need get the class name, where find image and transform image to digital 
        self.image_path = image_path    # each image path
        self.class_name = class_name    # image class name    
        self.transform = transform
        
    def __len__(self):
        # in torch dataset we need save the number of samples
        return len(self.image_path)
    
    def __getitem__(self, index):
        # each image path
        img_path = self.image_path[index]
        image = Image.open(img_path).convert('RGB')
        # basename will gets the name of that folder and dirname gets the folder that contains the image
        # one is folder and other is get the name
        label_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_name.index(label_name)
        if self.transform:
            image = self.transform(image)
        return image, label
    

    
def get_dataloader(data_dir, transform):
    # get the picture class from the dir this step is to separate and summary
    classes = get_classes(data_dir)
    """
    # get the path of each picture
    # gothrough every directory in train, x is the class name x[0] is current directory path
    # glob will get all .jpg in that directory
    # y is picture from that directory, loop to get all of them
    image_paths = [y for x in os.walk(data_dir) for y in glob(os.path.join(x[0], '*.jpg'))]
    # WE NEED SEPATRATE train data to train and validation to test overfitting
    train_paths, val_paths = train_test_split(image_paths, test_size=test_split, random_state=2)
    # transform picture to tensor
    # add data augmentation
    """
    image_path = [y for x in os.walk(data_dir) for y in glob(os.path.join(x[0], '*.jpg'))]
    data = ImageDataset(image_path, classes, transform)
    
    
    return data, classes
    

        
    