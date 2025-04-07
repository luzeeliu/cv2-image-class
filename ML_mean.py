import os
import cv2
from ML_SIFT_SVM import SIFT
from k_means_vocb import k_means
from Bow import Bow
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

acc = accuracy_score(y_test, y_pred)
cp = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test,y_pred)

print(f"accuracy:{acc}")
print(f"Classification Report:{cp}")
print(f"Confusion Matrix:{cm}")
