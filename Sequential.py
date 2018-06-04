import keras
from keras import Sequential
from keras.layers import Dense, Activation

import tensorflow as tf
from keras.optimizers import Adam

from PreprocessingData import training_data


print(tf.__version__)
print(keras.__version__)

model = Sequential([
    Dense(16, input_shape=(1,)),
    Activation('relu'),
    Dense(32),
    Activation('relu'),
    Dense(2),
    Activation('softmax'),
])

model.summary()

model.compile(
    Adam(lr=.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

train_samples, train_label = training_data()
one_hot_labels = keras.utils.to_categorical(train_label, num_classes=2)

model.fit(train_samples,
          one_hot_labels,
          # validation_split=0.1,
          batch_size=20,
          epochs=30,
          shuffle=True,
          verbose=1)

# predictions = model.predict(train_samples)
predictions = model.predict_classes(train_samples)

# my_result = zip(predictions, train_label)
#
# Success = 0
# Failure = 0
#
# for i in my_result:
#     if i[0] == i[1]:
#         Success += 1
#     else:
#         Failure += 1
#
# print(f"Success: {Success}, Failure {Failure}")

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


cm = confusion_matrix(train_label, predictions)


