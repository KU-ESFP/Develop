import numpy as np
import cv2
import os


# Absolute path to dataset
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
# print(os.path.abspath(os.path.join(parent_dir, os.pardir)))

# Datasets Path
train_dir = os.path.abspath(os.path.join(parent_dir, os.pardir)) + '/datasets/train'
test_dir = os.path.abspath(os.path.join(parent_dir, os.pardir)) + '/datasets/test'


# Class Classification
classes = ['angry', 'happy', 'neutral', 'sad', 'surprised']
#classes = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']


# Image Patch
x_train = []
y_train = []

PATCH_SIZE = 48

for idx, class_name in enumerate(classes):
    image_dir = os.path.join(train_dir, class_name)
    for image_name in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, image_name))
        x_train.append(image)
        y_train.append(idx)

x_train = np.array(x_train)/128 - 1
x_train = np.swapaxes(x_train, 1, 3)
y_train = np.array(y_train)
print('train data: ', x_train.shape)
print('train label: ', y_train.shape)

x_test = []
y_test = []

for idx, class_name in enumerate(classes):
    image_dir = os.path.join(test_dir, class_name)
    for image_name in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, image_name))
        x_test.append(image)
        y_test.append(idx)

x_test = np.array(x_test)/128 - 1
x_test = np.swapaxes(x_test, 1, 3)
y_test = np.array(y_test)
print('train data: ', x_test.shape)
print('train label: ', y_test.shape)


