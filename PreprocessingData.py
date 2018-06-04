from random import randint
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def training_data():
    train_labels = list()
    train_samples = list()

    for i in range(1000):
        random_younger = randint(13, 64)
        train_samples.append(random_younger)
        train_labels.append(0)

        random_older = randint(65, 100)
        train_samples.append(random_older)
        train_labels.append(1)

    for i in range(50):
        random_younger = randint(13, 64)
        train_samples.append(random_younger)
        train_labels.append(1)

        random_older = randint(65, 100)
        train_samples.append(random_older)
        train_labels.append(0)

    # numpy array is a requirement of keras for label and samples
    train_labels = np.array(train_labels)
    train_samples = np.array(train_samples)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

    return train_samples, train_labels


def testing_data():
    test_labels = list()
    test_samples = list()

    for i in range(1000):
        random_younger = randint(13, 64)
        test_samples.append(random_younger)
        test_labels.append(0)

        random_older = randint(65, 100)
        test_samples.append(random_older)
        test_labels.append(1)

    for i in range(50):
        random_younger = randint(13, 64)
        test_samples.append(random_younger)
        test_labels.append(1)

        random_older = randint(65, 100)
        test_samples.append(random_older)
        test_labels.append(0)

    # numpy array is a requirement of keras for label and samples
    test_labels = np.array(test_labels)
    test_samples = np.array(test_samples)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_samples = scaler.fit_transform(test_samples.reshape(-1,1))

    return test_samples, test_labels

