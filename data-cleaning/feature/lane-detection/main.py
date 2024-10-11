import copy
import cv2
import numpy as np
import math

from actions.ImageMasks import ImageMasks
from actions.SmoothingMethods import SmoothingMethods
from actions.ThresholdMethods import ThresholdMethods
from actions.EdgeDetectors import EdgeDetectors
from actions.LineMethods import LineMethods

WINDOW_NAME = "Lane Marking Detector"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) 

def get_image_grid(image_array, row_length = 3):
    x_count = len(image_array)
    image_rows = []
    white_image = np.ones(image_array[0].shape, dtype=image_array[0].dtype) * 255
    
    for i in range(0, x_count, row_length):
        row_images = image_array[i:i + row_length]
        if len(row_images) < row_length:
            row_images.extend([white_image] * (row_length - len(row_images)))

        row_stack = np.hstack(row_images)
        image_rows.append(row_stack)
    
    output_image = np.vstack(image_rows)
    return output_image

# get image
originalImage = cv2.imread('./lane_detection_7.jpeg', cv2.IMREAD_GRAYSCALE)
image = copy.deepcopy(originalImage)


def run_pipeline(image):
    # smoothing
    SmoothingMethods.GaussianBlur(image)

    # threshold for white
    ThresholdMethods.adaptiveThesholding(image)

    # Applying the Canny Edge filter 
    edge = EdgeDetectors.applyCanny(image)

    # Apply hough lines
    line_image = LineMethods.applyHoughLines(edge)
    hough = cv2.addWeighted(originalImage, 0.8, line_image, 1, 1)

    # build output image
    outputImage = get_image_grid([originalImage, image, edge, hough], row_length = 2)

    cv2.imshow(WINDOW_NAME, outputImage)

run_pipeline(image)

cv2.waitKey(0)