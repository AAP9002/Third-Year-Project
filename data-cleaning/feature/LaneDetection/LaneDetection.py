import copy
import cv2
import numpy as np

from .actions.ImageMasks import ImageMasks
from .actions.SmoothingMethods import SmoothingMethods
from .actions.ThresholdMethods import ThresholdMethods
from .actions.EdgeDetectors import EdgeDetectors
from .actions.LineMethods import LineMethods
from .utils.ImageStacker import get_image_grid

WINDOW_NAME = "Lane Marking Detector"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) 

class LaneDetection:
    def run_pipeline(coloured_image):
        whiteMasked = ImageMasks.maskByColour(coloured_image, np.array([0, 0, 150]), np.array([180, 50, 255]))
        image = copy.deepcopy(whiteMasked)
        black_and_white_image = cv2.cvtColor(coloured_image, cv2.COLOR_BGR2GRAY)

        # ImageMasks.maskImageAboveY(image,520)
        # ImageMasks.maskImageBelowY(image,880)

        # smoothing
        SmoothingMethods.applyClosing(image)
        closed = copy.deepcopy(image)

        # threshold for white
        ThresholdMethods.adaptiveThesholding(image)

        # Applying the Canny Edge filter 
        edge = EdgeDetectors.applyCanny(image)

        # Apply hough lines
        line_image = LineMethods.applyHoughLines(edge, minLineLength=100)
        hough = cv2.addWeighted(black_and_white_image, 0.8, line_image, 1, 1)

        # build output image
        outputImage = get_image_grid([black_and_white_image, whiteMasked,closed, image, edge, hough] , row_length = 2)
        return outputImage