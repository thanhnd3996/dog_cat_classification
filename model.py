from pretrained.smallvggnet import SmallVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from imutils import paths
import numpy as np
import argparse
import pickle
import random
import cv2
import os

# parameters
batch_size = 32
epochs = 100

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset of images")
parser.add_argument("-l", "--labelbin", required=True,
                    help="path to output label binarizer")
args = parser.parse_args()

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
image_paths = sorted(list(paths.list_images(args.dataset)))
random.seed(42)
random.shuffle(image_paths)

# loop over the input images
for image_path in image_paths:
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (64, 64))
    image = np.reshape(image, (64, 64, 1))
    data.append(image)
    label = image_path.split(os.path.sep)[-2]
    label = np.reshape(label, -1)
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print(labels)

# partition the data into training and validation
(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# convert the labels from integers to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# data generator
aug = ImageDataGenerator(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=20,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")

# init the model and optimizer
model = SmallVGGNet.build(width=64, height=64, depth=1, classes=len(lb.classes_))

# train the network
file_path = './checkpoints/model/h5'
checkpoints = ModelCheckpoint(file_path, save_best_only=True, verbose=1, monitor='val_acc', mode='max')
print("[INFO] training network...")
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adadelta",
              metrics=['accuracy'])
model.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size),
                    validation_data=(x_test, y_test),
                    steps_per_epoch=len(x_train) // batch_size,
                    epochs=epochs,
                    callbacks=[checkpoints])

# result
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# label binarizer to disk
f = open(args.labelbin, 'wb')
f.write(pickle.dumps(lb))
f.close()
