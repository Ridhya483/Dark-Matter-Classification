import numpy as np
import os
import cv2

DATADIR = "C:/Users/RIDZZ ZOLDYCK/Desktop/Dark_Matter_Classification/Dataset/TrainData"        # directory of the dataset of training data
CATEGORIES = ["darkMatter" , "noMatter"]
TESTDIR = "C:/Users/RIDZZ ZOLDYCK/Desktop/Dark_Matter_Classification/Dataset/TestData"         # directory of testing data
TEST_CATEGORY = ["testdark" , "testNoDark" ]

       

img_size = 150     # dimension / height and width of image 

training_data = []
test_data = []

# creates an image array and appends it to our training data 
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img) , cv2.IMREAD_GRAYSCALE)
            training_data.append([img_array , class_num])

# creates an image array and appends it to our test data 
def create_test_data():
    for category in TEST_CATEGORY:
        path = os.path.join(TESTDIR,category)
        class_num = TEST_CATEGORY.index(category)
        for img in os.listdir(path):
            img_array_test = cv2.imread(os.path.join(path,img) , cv2.IMREAD_GRAYSCALE)
            test_data.append([img_array_test , class_num])
    


create_training_data()    # function call for creation of training data
create_test_data()          # function call for creation of test data

print( " Number of Images in Dataset : ")
print(len(training_data))  # gives us number of images in the training data


import random
random.shuffle(training_data)   # shuffling of train data for better accuracy

print( " Labels after Shuffling : ")

for sample in training_data[:10]:    # to see if the data has been shuffled or not
    print(sample[1])


x = []
y = []

# we will now create 2 files
# one will store the features
# the other will store the label [ given during encoding ]
for features , label in training_data :   
    x.append(features)
    y.append(label)
    

x = np.array(x).reshape(-1,img_size,img_size,1)   # reshaping the features list into a numpy array


# saving our featues and label file 
import pickle

pickle_out = open("X.pickle" , "wb")
pickle.dump(x , pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle" , "wb")
pickle.dump(y , pickle_out)
pickle_out.close()

print( "Features and Labels have been saved " )


# now we do the same with test data
print("\n\nNumber of Images in test data")
print(len(test_data))


x_test = []
y_test = []


random.shuffle(test_data)

for sample in test_data[:10]:
    print(sample[1])


for features , labels in test_data:
    x_test.append(features)
    y_test.append(label)


x_test = np.array(x_test).reshape(-1,img_size,img_size,1)

pickle_out = open("X_TEST.pickle" , "wb")
pickle.dump(x_test , pickle_out)
pickle_out.close()

pickle_out = open("Y_TEST.pickle" , "wb")
pickle.dump(y_test , pickle_out)
pickle_out.close()

print( "\n\nFeatures and Labels for Test Data have been saved " )

    











