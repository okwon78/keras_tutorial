from os import getcwd, listdir
from os.path import isfile, join
from shutil import move

from keras.preprocessing.image import ImageDataGenerator

cwd = getcwd()
path = join(cwd, 'cats-and-dogs/base_train')

train_dog = join(cwd, 'cats-and-dogs/train/dog')
train_cat = join(cwd, 'cats-and-dogs/train/cat')

valid_dog = join(cwd, 'cats-and-dogs/valid/dog')
valid_cat = join(cwd, 'cats-and-dogs/valid/cat')

test_dog = join(cwd, 'cats-and-dogs/test/dog')
test_cat = join(cwd, 'cats-and-dogs/test/cat')

for image in listdir(path):
    image_path = join(path, image)

    if (not '.jpg' in image) or (not isfile(image_path)):
        continue

    if 'dog' in image:
        move(image_path, join(train_dog, image))

    if 'cat' in image:
        move(image_path, join(train_cat, image))

count_dog_image = 0

for image in listdir(train_dog):
    if 'dog' in image:
        count_dog_image += 1

    image_path = join(train_dog, image)
    if 10000 < count_dog_image < 11000:
        move(image_path, join(test_dog, image))

    if count_dog_image > 11000:
        move(image_path, join(valid_dog, image))

count_cat_image = 0

for image in listdir(train_cat):
    if 'cat' in image:
        count_cat_image += 1

    image_path = join(train_cat, image)
    if 10000 < count_cat_image < 11000:
        move(image_path, join(test_cat, image))

    if count_cat_image > 11000:
        move(image_path, join(valid_cat, image))

print(f'Dog image : {count_dog_image}')
print(f'Cat image : {count_cat_image}')

image_dir = join(getcwd(), 'cats-and-dogs')

train_path = join(image_dir, 'train')
valid_path = join(image_dir, 'valid')
test_path = join(image_dir, 'test')

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224, 224), classes=['dog', 'cat'],
                                                         batch_size=20)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224, 224), classes=['dog', 'cat'],
                                                         batch_size=5)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224, 224), classes=['dog', 'cat'],
                                                        batch_size=10)

