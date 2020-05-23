import numpy as np
import cv2
from utilities import *


# This function applies all elaboration steps to the image
def get_roi(raw_image):
    height, width = raw_image.shape

    points = np.array([[150, 350], [200, 275.0], [320.0, 275.0], [410, 275.0], [460, 350]], np.int32)
    points = points.reshape((-1, 1, 2))
    black_image = np.zeros((height, width, 1), np.uint8)

    polygonal_shape = cv2.fillPoly(black_image, [points], (255, 255, 255))
    colored_masked_road = cv2.bitwise_and(raw_image, raw_image, mask=polygonal_shape)

    # convert back to BRG/jaa
    result = cv2.cvtColor(colored_masked_road, cv2.COLOR_GRAY2BGR)
    # result = reshape(result)

    return result


# reshape to 640 x 480 frame size/jaa
def reshape(img):
    im = cv2.resize(img, (640, img.shape[0]))

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, 178, 177, 0, 0, cv2.BORDER_CONSTANT, value=color)

    return new_im
