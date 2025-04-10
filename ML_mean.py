import os
import cv2
from ML_SIFT_SVM import SIFT
from k_means_vocb import k_means
from Bow import Bow
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from evaluate import evaluate_model

train_path = os.path.join("dataset","train")
test_path = os.path.join("dataset","test")
num_cluster = 500

# get the SIFT from training dataset
train_descriptors = SIFT(train_path)
test_descriptors = SIFT(test_path)

#create the vocb
kmeans = k_means(train_descriptors, num_cluster)

#get bow of train and test
x_train, y_train = Bow(train_descriptors, kmeans, num_cluster)
x_test, y_test = Bow(test_descriptors, kmeans, num_cluster)

#scale train to standard
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#train the classificator
clf = SVC(kernel="linear", random_state= 42)
clf.fit(x_train, y_train)

#validation
y_pred = clf.predict(x_test)

evaluate_model(y_test,y_pred,"Machine Learning")
