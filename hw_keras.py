''' Operations for keras
''' 
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import cifar10, imdb, reuters, mnist
from keras.preprocessing import image as keras_image
# Load specific models
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
# Load Xception
from keras.applications.xception import Xception

# External image datasets
from pathlib import Path
from PIL import Image

nnet = Sequential()
nnet.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
nnet.add(Conv2D(32, (3, 3), activation='relu'))
nnet.add(MaxPooling2D(pool_size=(2, 2)))
nnet.add(Dropout(0.25))

nnet.add(Flatten())
nnet.add(Dense(256, activation='relu'))
nnet.add(Dropout(0.5))
nnet.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

nnet.compile(loss='categorical_crossentropy',
             optimizer=sgd,
             metrics=['accuracy'])

# CIFAR-10 class name
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

# Load CIFAR dataset
def train_nnet(nnet, feats, labels):
    X_train = feats.reshape((feats.shape[0], 3, 32, 32))
    X_train = X_train.swapaxes(1, 2).swapaxes(2, 3)
    y_train = np.zeros((labels.shape[0], len(classes)))
    for i, kls in enumerate(labels):
        y_train[i][kls-1] = 1

    nnet.fit(X_train, y_train, epochs=5, batch_size=64)
    loss_and_metrics = nnet.evaluate(X_train, y, batch_size=64)


nnet_res50 = ResNet50(weights='imagenet')

root_fpath = Path.home() / 'local' / 'data' 
img_fpath = root_fpath / 'images' / 'cat.jpg'
image = keras_image.load_img(img_fpath, target_size=(224, 224))
x = keras_image.img_to_array(image)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = nnet_res50.predict(x)
print(decode_predictions(preds, top=5)[0])

# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
