import cv2
import numpy as np

def compute_bow_histogram(descriptor, kmeans, num_cluster = 200):
    # get vs vocab and get shape
    histogram =  np.zeros(num_cluster)
    if descriptor is not None:
        # predict whcih cluster each SIFT descriptor belong to 
        # count how many items each cluster appears create histogram
        words = kmeans.predict(descriptor)
        for word in words:
            histogram[word] += 1
    
    return histogram

def Bow(descriptors, kmeans, num_cluster = 200):
    x_train = []
    y_train = []
    
    for label, list in descriptors.items():
        for descriptor in list:
            histogram = compute_bow_histogram(descriptor, kmeans, num_cluster)
            x_train.append(histogram)
            y_train.append(label)
    
    return x_train, y_train