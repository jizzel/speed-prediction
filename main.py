import cv2
import os, time, sys, shutil
import numpy as np
import matplotlib.pyplot as plt
from enhance_image import *

from optical_flow import convertToOpticalFlow

PATH_DATA_FOLDER = './data/'

PATH_TRAIN_LABEL_PREPROCESSED = PATH_DATA_FOLDER + 'train_preprocessed.txt'

PATH_TRAIN_LABEL = PATH_DATA_FOLDER + 'train.txt'
PATH_TRAIN_VIDEO = PATH_DATA_FOLDER + 'train.mp4'
PATH_TRAIN_FLOW_VIDEO = PATH_DATA_FOLDER + 'flow_train.mp4'
PATH_TRAIN_IMAGES_FOLDER = PATH_DATA_FOLDER + 'train_images/'
PATH_TRAIN_IMAGES_FLOW_FOLDER = PATH_DATA_FOLDER + 'train_images_flow/'

PATH_TEST_LABEL = PATH_DATA_FOLDER + 'test.txt'
PATH_TEST_VIDEO = PATH_DATA_FOLDER + 'test.mp4'
PATH_TEST_FLOW_VIDEO = PATH_DATA_FOLDER + 'flow_test.mp4'
PATH_TEST_IMAGES_FOLDER = PATH_DATA_FOLDER + 'test_images/'
PATH_TEST_IMAGES_FLOW_FOLDER = PATH_DATA_FOLDER + 'test_images_flow/'


def preprocess_data(video_input_path, flow_video_output_path, image_folder_path, flow_image_folder_path, type):
    if os.path.exists(image_folder_path):
        shutil.rmtree(image_folder_path)
    os.makedirs(image_folder_path)
    if os.path.exists(flow_image_folder_path):
        shutil.rmtree(flow_image_folder_path)
    os.makedirs(flow_image_folder_path)

    print("Converting video to optical flow for: ", video_input_path)

    video_reader = cv2.VideoCapture(video_input_path)
    # print('video path: ', video_input_path)
    # print('video reader: ', video_reader)

    num_frames = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_size = (int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # fourcc = 0x763470d
    video_writer = cv2.VideoWriter(flow_video_output_path, fourcc, fps, frame_size)

    t1 = time.time()
    ret, prev_frame = video_reader.read()
    hsv = np.zeros_like(prev_frame)

    image_path_out = os.path.join(image_folder_path, str(0) + '.png')
    cv2.imwrite(image_path_out, prev_frame)

    print('Starting now...')
    prev_frame = elaborateImage(prev_frame)

    count = 1
    while True:
        ret, next_frame = video_reader.read()
        if next_frame is None:
            break
        # print('before dimension: ', prev_frame.shape)
        # print('before frame: ', prev_frame)

        # prev_frame = elaborateImage(prev_frame)
        next_frame = elaborateImage(next_frame)

        # print('after frame: ', prev_frame)
        # print('after dimension: ', prev_frame.shape)

        bgr_flow = convertToOpticalFlow(prev_frame, next_frame)

        image_path_out = os.path.join(image_folder_path, str(count) + '.png')
        flow_image_path_out = os.path.join(flow_image_folder_path, str(count) + '.png')

        cv2.imwrite(image_path_out, next_frame)
        cv2.imwrite(flow_image_path_out, bgr_flow)

        video_writer.write(bgr_flow)
        # print('Counting ', count)
        sys.stdout.write('\rprocessed frames: %d' % count)

        prev_frame = next_frame
        count += 1

    t2 = time.time()
    video_reader.release()
    video_writer.release()
    print(' Conversion completed !')
    print(' Time Taken:', (t2 - t1), 'seconds')

    return


if __name__ == '__main__':
    '''PREPROCESS DATA DOES 3 THINGS:
        1. Convert video to optical flow and save their respective images
        2. Augment image and optical flow data by Inverting them horizontally'''  ## NOW DONE IN TRAIN_MODEL.PY ITSELF IN GENERATOR DATA

    preprocess_data(PATH_TRAIN_VIDEO, PATH_TRAIN_FLOW_VIDEO, PATH_TRAIN_IMAGES_FOLDER, PATH_TRAIN_IMAGES_FLOW_FOLDER,
                    type='train')
    preprocess_data(PATH_TEST_VIDEO, PATH_TEST_FLOW_VIDEO, PATH_TEST_IMAGES_FOLDER, PATH_TEST_IMAGES_FLOW_FOLDER,
                    type='test')
