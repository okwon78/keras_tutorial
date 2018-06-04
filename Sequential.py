import keras
from keras import Sequential
from keras.layers import Dense, Activation

import tensorflow as tf
from keras.models import load_model, model_from_json
from keras.optimizers import Adam

from PreprocessingData import training_data, testing_data

print(tf.__version__)
print(keras.__version__)

try:
    model = load_model('my_model.h5')
    print(model.get_weights())
except OSError:
    train_samples, train_label = training_data()
    one_hot_labels = keras.utils.to_categorical(train_label, num_classes=2)

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

    model.fit(train_samples,
              one_hot_labels,
              # validation_split=0.1,
              batch_size=20,
              epochs=30,
              shuffle=True,
              verbose=1)

    model.save('my_model.h5')


json_string = model.to_json()
model_arch = model_from_json(json_string)

test_samples, test_label = testing_data()
one_hot_labels = keras.utils.to_categorical(test_label, num_classes=2)

score = model.evaluate(test_samples, one_hot_labels, batch_size=10)

print('\n', f'{model.metrics_names[0]}: {score[0]}')
print(f'{model.metrics_names[1]}: {score[1]}', '\n')

# predictions = model.predict(train_samples)
predictions = model.predict_classes(test_samples)

Success = 0
Failure = 0

for index, val in enumerate(predictions) :
    if val == test_label[index]:
        Success += 1
    else:
        Failure += 1

print(f"Success: {Success}, Failure {Failure}")
