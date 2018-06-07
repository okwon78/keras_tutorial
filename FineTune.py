from datetime import time

import keras
from keras import Model
from keras.callbacks import TensorBoard

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import load_model
from keras.optimizers import Adam

from CNN import draw_result
from ImgPrepare import valid_batches, train_batches, test_batches


def get_model():

    try:
        return load_model('find_tune.h5')
    except IOError:
        inception_v3 = keras.applications.inception_v3.InceptionV3(
            include_top=False,
            input_shape=(224, 224, 3),
            weights='imagenet')

        for layer in inception_v3.layers:
            layer.trainable = False

        x = inception_v3.output
        x = GlobalAveragePooling2D(name='my_avg_pool')(x)
        predictions = Dense(2, activation="softmax")(x)

        return Model(input=inception_v3.input, output=predictions)


if __name__ == '__main__':
    model_final = get_model()

    model_final.summary()

    model_final.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    model_final.fit_generator(train_batches,
                              validation_data=valid_batches,
                              validation_steps=5,
                              steps_per_epoch=10, epochs=100,
                              verbose=2, callbacks=[tensorboard])

    model_final.save('find_tune.h5')

    draw_result(model_final, test_batches)

