# from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from imutils import paths
import argparse
import pickle
import random
import cv2
import os

# parameters
classes = 2
batch_size = 32
pool_size = (2, 2)
kernel_size = (3, 3)
epochs = 100

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset of images")
# parser.add_argument("-l", "--labelbin", required=True,
#                     help="path to output label binarizer")
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
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1399, 1399))
    data.append(image)
    label = image_path.split(os.path.sep)[-2]
    labels.append(label)

# scale the raw pixel intensities to the range [0,1]
# data = np.array(data, dtype='float') / 255
# labels = np.array(labels)

# partition the data into training and validation
(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.3, random_state=42)

# convert the labels from integers to vectors
# lb = LabelBinarizer()
# y_train = lb.fit_transform(y_train)
# y_test = lb.transform(y_test)

# pre-processing
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
# init the model and optimizer
base_model = ResNet50(weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
preds = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=preds)

# train the network
file_path = './checkpoints/model/h5'
checkpoints = ModelCheckpoint(file_path, save_best_only=True, verbose=1, monitor='val_acc', mode='max')
print("[INFO] training network...")
model.compile(loss="categorical_crossentropy",
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
# f = open(args.labelbin, 'wb')
# f.write(pickle.dumps(lb))
# f.close()
