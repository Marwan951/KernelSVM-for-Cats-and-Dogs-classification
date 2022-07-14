import numpy as np
import timeit
import os  # read the file
import cv2  # read images
from skimage.feature import hog  # extracted features of each image
from sklearn import svm  # to classify the data
from sklearn.model_selection import train_test_split  # split and shuffle data

# get all data from Train file
data_File_path = r'C:\3rd_year\second term\pattern recognition\labs\lab7\dogs_vs_cats\train'
images = os.listdir(data_File_path)
X_train = []
Y_train = []
x_test = []
y_test = []
new_image = []
i = 0

# get 1100 cats and 1100 dogs
while i < 1101:
    if images[i].__contains__("cat"):
        new_image.append(images[i])
    #  print("Image numer", i, "is", new_image[-1])
    i = i + 1

g = 12501
b = 12501 + 1100
while g < b:
    if images[g].__contains__("dog"):
        new_image.append(images[g])
    # print("Image numer", g, "is", new_image[-1])
    g = g + 1

# //////////////////////////////////////////////////////////////////////////////////////////////////////
# 2200 row for -> 2000 train& 200test
images = new_image
# Shuffle data
img, _, _, _ = train_test_split(images, np.zeros(len(images)), test_size=0.00000000000001, random_state=25,
                                shuffle=True)
# //////////////////////////////////////////////////////////////////////////////////////////////////////

# img  -> random data that we will work with
print(len(img), "  Data row : that we will work with ", "\n", img)
# //////////////////////////////////////////////////////////////////////////////////////////////////////

# iterator to get trained data [2000] into -> X
print(" 2000 trained rows : that we will work with ", "\n")
i = 0
while i < 2000:

    img_array0 = cv2.imread(os.path.join(data_File_path, img[i]), cv2.IMREAD_GRAYSCALE)
    img_array0 = cv2.resize(img_array0, (128, 64))
    fd0, hog_image0 = hog(img_array0, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    X_train.append(fd0)
    if img[i].__contains__("cat"):
        Y_train.append(0)
    else:
        Y_train.append(1)
    print(X_train[-1], "    ->    ", Y_train[-1])
    i = i + 1
print(len(X_train))

print(" 200 trained rows : that we will work with ", "\n")
# iterator to get tested data [200] into -> x_test
e = 0
while e < 200:
    # to start getting data after the train rows
    w = len(X_train)
    img_array1 = cv2.imread(os.path.join(data_File_path, img[w]), cv2.IMREAD_GRAYSCALE)
    img_array1 = cv2.resize(img_array1, (128, 64))
    fd1, hog_image1 = hog(img_array1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

    x_test.append(fd1)
    if img[e].__contains__("cat"):
        y_test.append(0)
    else:
        y_test.append(1)

    print("x_test Number ", e, " = ", x_test[e], "     has label value =    ", y_test[-1])
    w = w + 1
    e = e + 1
print(len(x_test))
# //////////////////////////////////////////////////////////////////////////////////////////////////////


# After Processing data :-
#  X -> 2000 rows [1000 -> cats & 1000 -> dogs] (contains the extracted features from images )
#  Y -> 2000 rows [1000 -> cats & 1000 -> dogs] (contains the label for each image )
# x_test -> 200 rows [100 -> cats & 100 -> dogs] (contains the extracted features from images )
# y_test -> 200 rows [100 -> cats & 100 -> dogs] (contains the label for each image )
# //////////////////////////////////////////////////////////////////////////////////////////////////////

# Classify It Using SVM Model

# regularization parameter
C = 0.1
# 1
tic1 = timeit.default_timer()
svc1 = svm.LinearSVC(C=C).fit(X_train, Y_train)
toc1 = timeit.default_timer()
print("svc time         :", toc1 - tic1)

# 2 sigmoid
tic2 = timeit.default_timer()
sigmoid_svc = svm.SVC(kernel='sigmoid', degree=3, C=C).fit(X_train, Y_train)
toc2 = timeit.default_timer()
print("sigmoid_svc time :", toc2 - tic2)

# 3 rbf
tic3 = timeit.default_timer()
Rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(X_train, Y_train)
toc3 = timeit.default_timer()
print("Rbf_svc time     :", toc3 - tic3)

# 4 linear
tic4 = timeit.default_timer()
linsvc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)
toc4 = timeit.default_timer()
print("linsvc time      :", toc4 - tic4)

# 5 poly
tic5 = timeit.default_timer()
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, Y_train)
toc5 = timeit.default_timer()
print("pol time         :", toc5 - tic5)
# //////////////////////////////////////////////////////////////////////////////////////////////////////

# After Training the model calc train and test error for model with each kernel function to get the best one

print(" test svc Accuracy       :", (svc1.score(x_test, y_test)) * 100, "%")
print(" test lin_svc Accuracy   :", (linsvc.score(x_test, y_test)) * 100, "%")
print(" test rbf_svc Accuracy   :", (Rbf_svc.score(x_test, y_test)) * 100, "%")
print(" test poly_svc Accuracy  :", (poly_svc.score(x_test, y_test)) * 100, "%")
print(" test sig_svc Accuracy   :", (sigmoid_svc.score(x_test, y_test)) * 100, "%")

print("train svc accuracy     :", svc1.score(X_train, Y_train) * 100, "%")
print("train lin_svc accuracy :", linsvc.score(X_train, Y_train) * 100, "%")
print("train rbf_svc accuracy :", Rbf_svc.score(X_train, Y_train) * 100, "%")
print("train poly_svc accuracy:", poly_svc.score(X_train, Y_train) * 100, "%")
print("train sig_svc accuracy :", sigmoid_svc.score(X_train, Y_train) * 100, "%")
