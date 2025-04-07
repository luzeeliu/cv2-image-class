import cv2
import numpy as np
import os 
from sklearn.cluster import MiniBatchKMeans

"""_summary_
for bag of visual words, use k-means clustering to build a visual vocabulary just like vocb in LSTM but is cv
from all descriptors
"""

def k_means(descriptors, num_cluster = 200):
    all_descriptors = []
    
    # cluster all descriptor into one big array
    for des_list in descriptors.values():
        for i in des_list:
            all_descriptors.append(i)
    # stack into a (N,128) array (keypoint, vector)
    # just like zip
    all_descriptors = np.vstack(all_descriptors)
    
    
    # configure the kmean manipulate 
    kmean = MiniBatchKMeans(n_clusters=num_cluster, random_state= 42)
    # build visual vocabulary
    kmean.fit(all_descriptors)
    
    return kmean
    
    
    
    
    
    
    
    