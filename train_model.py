# from model2 import CNNModel
# from model3 import CNNModel
import time

from model import CNNModel

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.callbacks import EarlyStopping, ModelCheckpoint

from optical_flow import convertToOpticalFlow

PATH_DATA_FOLDER = './data/'
PATH_TRAIN_LABEL = PATH_DATA_FOLDER + 'train.txt'
PATH_TRAIN_IMAGES_FOLDER = PATH_DATA_FOLDER + 'train_images/'
PATH_TRAIN_IMAGES_FLOW_FOLDER = PATH_DATA_FOLDER + 'train_images_flow/'

TYPE_FLOW_PRECOMPUTED = 0
TYPE_ORIGINAL = 1

BATCH_SIZE = 128
EPOCH = 50

MODEL_NAME = 'CNNModel_flow'


# MODEL_NAME = 'CNNModel_combined'


def prepareData(labels_path, images_path, flow_images_path, type=TYPE_FLOW_PRECOMPUTED):
    num_train_labels = 0
    train_labels = []
    train_images_pair_paths = []

    with open(labels_path) as txt_file:
        labels_string = txt_file.read().split()

        for i in range(4, len(labels_string)):
            speed = float(labels_string[i])
            train_labels.append(speed)
            num_train_labels += 1

            if type == TYPE_FLOW_PRECOMPUTED:
                # Combine original and pre computed optical flow
                train_images_pair_paths.append((os.getcwd() + images_path[1:] + str(i) + '.png',
                                                os.getcwd() + flow_images_path[1:] + str(i - 3) + '.png',
                                                os.getcwd() + flow_images_path[1:] + str(i - 2) + '.png',
                                                os.getcwd() + flow_images_path[1:] + str(i - 1) + '.png',
                                                os.getcwd() + flow_images_path[1:] + str(i) + '.png'))
            else:
                # Combine 2 consecutive frames and calculate optical flow
                train_images_pair_paths.append((os.getcwd() + images_path[1:] + str(i - 1) + '.png',
                                                os.getcwd() + images_path[1:] + str(i) + '.png'))

    return train_images_pair_paths, train_labels, num_train_labels


def generatorData(samples, batch_size=32, type=TYPE_FLOW_PRECOMPUTED):
    num_samples = len(samples)

    while 1:  # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:

                combined_image = None
                flow_image_bgr = None

                if type == TYPE_FLOW_PRECOMPUTED:

                    curr_image_path, flow_image_path1, flow_image_path2, flow_image_path3, flow_image_path4 = imagePath
                    path1 = cv2.imread(flow_image_path1)
                    path2 = cv2.imread(flow_image_path2)
                    path3 = cv2.imread(flow_image_path3)
                    path4 = cv2.imread(flow_image_path4)

                    a = (path1 + path2 + path3 + path4)
                    flow_image_bgr = a / 4

                    curr_image = cv2.imread(curr_image_path)
                    curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)

                else:
                    prev_image_path, curr_image_path = imagePath
                    prev_image = cv2.imread(prev_image_path)
                    curr_image = cv2.imread(curr_image_path)
                    flow_image_bgr = convertToOpticalFlow(prev_image, curr_image)
                    curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)

                combined_image = 0.1 * curr_image + flow_image_bgr
                # CHOOSE IF WE WANT TO TEST WITH ONLY OPTICAL FLOW OR A COMBINATION OF VIDEO AND OPTICAL FLOW
                combined_image = flow_image_bgr

                combined_image = cv2.normalize(combined_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX,
                                               dtype=cv2.CV_32F)
                combined_image = cv2.resize(combined_image, (0, 0), fx=0.5, fy=0.5)

                images.append(combined_image)
                angles.append(measurement)

                # AUGMENTING DATA
                # Flipping image, correcting measurement and measurement

                images.append(cv2.flip(combined_image, 1))
                angles.append(measurement)

            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)


if __name__ == '__main__':
    type_ = TYPE_FLOW_PRECOMPUTED  # optical flow pre computed
    # type_ = TYPE_ORIGINAL

    train_images_pair_paths, train_labels, labels_count = prepareData(PATH_TRAIN_LABEL, PATH_TRAIN_IMAGES_FOLDER,
                                                                      PATH_TRAIN_IMAGES_FLOW_FOLDER, type=type_)

    samples = list(zip(train_images_pair_paths, train_labels))
    train_samples, validation_samples = train_test_split(samples, test_size=0.15)

    print('Total Images: {}'.format(len(train_images_pair_paths)))
    print('Train samples: {}'.format(len(train_samples)))
    print('Validation samples: {}'.format(len(validation_samples)))
    print('Number of labels: {}'.format(labels_count))

    training_generator = generatorData(train_samples, batch_size=BATCH_SIZE, type=type_)
    validation_generator = generatorData(validation_samples, batch_size=BATCH_SIZE, type=type_)

    print('Training model...')
    t1 = time.time()

    model = CNNModel()

    callbacks = [EarlyStopping(monitor='val_loss', patience=7, mode='auto'),
                 ModelCheckpoint(filepath='best' + MODEL_NAME + '.h5', monitor='val_loss', save_best_only=True, mode='auto')]

    history_object = model.fit_generator(
        training_generator,
        len(train_samples) // BATCH_SIZE,
        epochs=EPOCH,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=len(validation_samples) // BATCH_SIZE,
        class_weight=None,
        workers=1,
        initial_epoch=0,
        use_multiprocessing=False,
        max_queue_size=10)

    t2 = time.time()
    print('Training model complete...')
    print(' Time Taken:', (t2 - t1)/60, 'minutes')

    print('Loss: ')
    print(history_object.history['loss'])
    print('Validation Loss: ')
    print(history_object.history['val_loss'])

    print('Accuracy: ')
    print(history_object.history['accuracy'])
    print('Validation Loss: ')
    print(history_object.history['val_accuracy'])

    plt.figure(figsize=[10, 8])
    plt.plot(np.arange(1, len(history_object.history['loss']) + 1), history_object.history['loss'], 'r', linewidth=3.0)
    plt.plot(np.arange(1, len(history_object.history['val_loss']) + 1), history_object.history['val_loss'], 'b',
             linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.show()
    plt.savefig('graph.png')

    # Plot training & validation accuracy values
    plt.plot(np.arange(1, len(history_object.history['accuracy']) + 1), history_object.history['accuracy'], 'r', linewidth=3.0)
    plt.plot(np.arange(1, len(history_object.history['val_accuracy']) + 1), history_object.history['val_accuracy'], 'b',
             linewidth=3.0)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig('graph_2.png')
