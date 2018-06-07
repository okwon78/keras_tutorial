import itertools
import sys
import matplotlib.pyplot as plt
import numpy as np

from time import time

from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Activation, Flatten, Dense
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix

from ImgPrepare import test_batches, valid_batches


def plots(ims, figsize=(12, 6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if ims.shape[-1] != 3:
            ims = ims.transpose((0, 2, 3, 1))
    f = plt.figure(figsize=figsize)
    cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i + 1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
    plt.show()


# imgs, labels = next(train_batches)
# plots(imgs, titles=labels)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


import tensorflow as tf
import keras

print(tf.__version__)
print(keras.__version__)
print(sys.version)


def get_cnn_model():
    try:
        return load_model('my_cnn.h5')
    except IOError:
        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), name="conv1"))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), name="conv2"))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(2, name='fc'))
        model.add(Activation('softmax'))

        return model


def draw_result(model, test_bathes):
    predictions = model.predict_generator(test_batches, steps=1)
    predictions = predictions[:, 0]
    print(predictions)

    test_imgs, test_labels = next(test_batches)
    test_labels = test_labels[:, 0]
    print(test_labels)

    plots(test_imgs, titles=test_labels)

    cm = confusion_matrix(test_labels, predictions)

    cm_plot_labels = ['cat', 'dog']
    plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


if __name__ == '__main__':
    model = get_cnn_model()

    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    model.fit_generator(train_batches,
                        validation_data=valid_batches,
                        validation_steps=5,
                        steps_per_epoch=10, epochs=5,
                        verbose=2, callbacks=[tensorboard])

    model.save('my_cnn.h5')

    model.summary()

    draw_result(model=model, test_bathes=test_batches)
