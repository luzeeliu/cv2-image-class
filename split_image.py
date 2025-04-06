import os
import random
import shutil
#os is used for folder manipulate
#shutil is used for picture manipulate
#set random seed
random.seed(42)

#the path of each dataset
data_path = 'data/Aerial_Landscapes'
train_dir = 'dataset/train'
test_dir = 'dataset/test'

#use os to make folder, exist_ok is for when use it befor create
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

#process each class folder
#os.listdir is loop each folder in dir
for class_folder in os.listdir(data_path):
    #open each class folder
    class_path = os.path.join(data_path, class_folder)
    if not os.path.isdir(class_path):
        continue
    
    #create class subfolder in train and test directories
    os.makedirs(os.path.join(train_dir, class_folder), exist_ok= True)
    os.makedirs(os.path.join(test_dir,class_folder), exist_ok=True)
    
    #class_forder is class folder name in data folder in original folder
    #class_path is we open the folder of each class in original folder
    
    
    #after create folder we need manipulate the picture 
    #get pictures form folder and random them then select first 80% as training 
    images = [i for i in os.listdir(class_path) if i.endswith('.jpg')]
    random.shuffle(images)
    
    split_id = int(len(images) * 0.8)
    train_images = images[:split_id]
    test_images = images[split_id:]
    
    #move image to new folder if want to copy it use shutil.copy
    for img in train_images:
        #new i know os.join can have three parameter
        shutil.move(os.path.join(class_path, img), os.path.join(train_dir, class_folder, img))
    
    for img in test_images:
        shutil.move(os.path.join(class_path, img), os.path.join(test_dir, class_folder, img))
        
print("data split complete")
    
    
    

