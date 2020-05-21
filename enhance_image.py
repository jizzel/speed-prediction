import numpy as np
import cv2
from utilities import *


# This function is able to highlight the contours of road lane markings using color thresholding and canny edge
def highlightRoadLaneMarkings(newFrame):
    # Adjusting brightness and contrast
    new_frame_adjusted = apply_brightness_contrast(newFrame, 100, 100)

    # Threshold so that only yellow and white are kept. Result is greyscale
    new_frame_threshold = thresholdWhiteAndYellow(new_frame_adjusted)

    # Apply Gaussian blur to reduce noise
    new_frame_blurred = cv2.GaussianBlur(new_frame_threshold, (5, 5), 0)

    # Applying canny edge detection
    newFrameEdges = cv2.Canny(new_frame_blurred, 100, 200)
    # cv2.imshow('hhh', newFrameEdges)
    # cv2.waitKey(10000)

    # Cutting a region of interest
    height, width = newFrameEdges.shape
    # Creating white polygonal shape on black image
    bottom_left = [0, height - 130]
    top_left = [width / 3 + 40, height / 2]
    top_right = [width / 3 * 2 - 40, height / 2]
    bottom_right = [width, height - 130]
    pts = np.array([bottom_left, top_left, top_right, bottom_right], np.int32)
    pts = pts.reshape((-1, 1, 2))
    black_image = np.zeros((height, width, 1), np.uint8)
    polygonal_shape = cv2.fillPoly(black_image, [pts], (255, 255, 255))
    # Doing AND operation with newFrameEdges
    new_frame_roi = cv2.bitwise_and(newFrameEdges, newFrameEdges, mask=polygonal_shape)

    return new_frame_roi


# This function applies all elaboration steps to the image
def elaborateImage(newFrame):
    # Drawing road from original frame
    new_frame_adjusted = apply_brightness_contrast(newFrame, 30, 15)
    new_frame_grey = cv2.cvtColor(new_frame_adjusted, cv2.COLOR_BGR2GRAY)
    height, width = new_frame_grey.shape
    bottom_left = [0, height - 130]
    top_left = [0, height / 2 + 10]
    top_center = [width / 2, height / 2 - 15]
    top_right = [width, height / 2 + 10]
    bottom_right = [width, height - 130]
    pts = np.array([bottom_left, top_left, top_center, top_right, bottom_right], np.int32)
    pts = pts.reshape((-1, 1, 2))
    black_image = np.zeros((height, width, 1), np.uint8)

    polygonal_shape = cv2.fillPoly(black_image, [pts], (255, 255, 255))
    colored_masked_road = cv2.bitwise_and(new_frame_grey, new_frame_grey, mask=polygonal_shape)
    new_frame_roi = highlightRoadLaneMarkings(newFrame)
    new_frame_mask_and_road = cv2.add(colored_masked_road, new_frame_roi)  # Adding canny edge overlay to highlight the lane markers

    # Cutting image basing on mask size
    # result = cutTopAndBottom(coloredMaskedRoad, int(height / 2 - 15), int(height - 130))
    result = cutTopAndBottom(new_frame_mask_and_road, int(height / 2 - 15), int(height - 130))

    # convert back to BRG/jaa
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    result = reshape(result)

    return result


# reshape to 640 x 480 frame size/jaa
def reshape(img):
    im = cv2.resize(img, (640, img.shape[0]))

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, 178, 177, 0, 0, cv2.BORDER_CONSTANT, value=color)

    return new_im
