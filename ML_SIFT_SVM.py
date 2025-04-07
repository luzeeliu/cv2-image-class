import cv2
import os 

path_train = os.path.join("dataset","train")
def SIFT(path_train):
    # create sift
    sift = cv2.SIFT_create()
    #create dictionary to return result
    # the key is the feature class and value is the descriptor of each feature
    sift_features = {}

    #loop the train folder
    for class_folder in os.listdir(path_train):
        #class_folder is class name not open the folder
        # we need open class first
        class_path = os.path.join(path_train, class_folder)
        if not os.path.isdir(class_path):
            continue
        # store descriptors as value in result
        descriptors_list = []
        
    # loop through imaghe in this class
        for img_file in os.listdir(class_path):
            if img_file.endswith('.jpg'):
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                keypoints, descriptors = sift.detectAndCompute(img_gray, None)
                
                if descriptors is not None:
                    descriptors_list.append(descriptors)
        sift_features[class_folder] = descriptors_list

    print("SIFT feature extract completed")
    return sift_features