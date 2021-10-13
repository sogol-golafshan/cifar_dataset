from keras.utils import np_utils
from keras.datasets.cifar10 import load_data
import seaborn as sns
from keras.initializers import RandomNormal  # or xaiver/Hae normilization
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D,Dropout,GlobalMaxPooling2D,Input
from keras.applications.vgg16 import VGG16


def plt_dynamic(x, vy, ty, colors=['b']):
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('epoch');
    ax.set_ylabel('Categorical Crossentropy Loss')

    ax.plot(x, vy, 'b', label="Validation Loss")
    ax.plot(x, ty, 'r', label="Train Loss")
    ax.legend()
    plt.grid()
    fig.canvas.draw()


def prepare_data():
    print('Preparing Data...')
    (X_train, y_train), (X_test, y_test) = load_data()
    Y_train = np_utils.to_categorical(y_train, 10)  # one hot incoding
    Y_test = np_utils.to_categorical(y_test, 10)  # one hot incoding

    return X_train, Y_train, X_test, Y_test


def create_model(input_dim, output_dim):
    model = VGG16(include_top=False,input_shape=(32,32,3))
    print('Model loaded.')

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(model)
    top_model.add(Flatten())
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(10, activation='softmax'))
    # add the model on top of the convolutional base
    #model.add(top_model)


    return top_model


def train(model, X_train, y_train, X_val, y_val, epoch_number, batch_size, optimizer, datagen):
    print('Start train...')
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch_number, verbose=1,
    #                    validation_data=(X_val, y_val))
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), epochs=epoch_number, verbose=1,
                        validation_data=(X_val, y_val))
    x = list(range(1, nb_epoch + 1))
    vy = history.history['val_loss']
    ty = history.history['loss']
    return model, x, vy, ty


def eval(model, X_test, Y_test):
    print('Start Eval...')
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
    argparser.add_argument('--batch_size', type=int, help='batch size', default=32)
    argparser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
    argparser.add_argument('--optimizer', help='valid optimizers are adam/sgd', default="adam")

    args = argparser.parse_args()

    X_train, y_train, X_test, y_test = prepare_data()

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    output_dim = 10  # equal to number of classes
    input_dim = X_train.shape[1]
    batch_size = args.batch_size
    nb_epoch = args.epoch
    optimizer = args.optimizer
    learning_rate = args.lr
    model = create_model(input_dim, output_dim)
    model, x, vy, ty = train(model, X_train, y_train, X_test, y_test, nb_epoch, batch_size, optimizer, datagen)
    eval(model, X_test, y_test)
    plt_dynamic(x, vy, ty)