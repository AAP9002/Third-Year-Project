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
        # SmoothingMethods.GaussianBlur(coloured_image, (25, 25))

        whiteMasked = ImageMasks.maskByColour(coloured_image, np.array([0, 0, 150]), np.array([180, 50, 255]))
        orangeMasked = ImageMasks.maskByColour(coloured_image, np.array([3, 90, 127]), np.array([33, 190, 227]))
        colourMasked = cv2.bitwise_or(whiteMasked, orangeMasked)

        image = copy.deepcopy(colourMasked)
        black_and_white_image = cv2.cvtColor(coloured_image, cv2.COLOR_BGR2GRAY)

        # threshold sky  
        sky_y = ImageMasks.get_sky_y_axis(black_and_white_image)
        ImageMasks.maskImageAboveY(image, sky_y)
        # draw skyline
        cv2.line(image, (0, sky_y), (image.shape[1], sky_y), (0, 0, 255), 2)

        # ImageMasks.maskImageAboveY(image,520)
        # ImageMasks.maskImageBelowY(image,880)

        # smoothing
        SmoothingMethods.applyClosing(image)
        closed = copy.deepcopy(image)

        # threshold for white
        ThresholdMethods.adaptiveThesholding(image)

        # compare against standard adaptive thresholding
        standard_adaptive = copy.deepcopy(black_and_white_image)
        ThresholdMethods.adaptiveThesholding(standard_adaptive)
        ImageMasks.maskImageAboveY(standard_adaptive, sky_y)


        # Applying the Canny Edge filter 
        edge = EdgeDetectors.applyCanny(image)
        standard_adaptive_edge = EdgeDetectors.applyCanny(standard_adaptive)

        # Apply hough lines
        line_image = LineMethods.applyHoughLines(edge, minLineLength=50, maxLineGap=10)
        hough = cv2.addWeighted(black_and_white_image, 0.8, line_image, 1, 1)

        standard_adaptive_line_image = LineMethods.applyHoughLines(standard_adaptive_edge, minLineLength=50, maxLineGap=10)
        standard_adaptive_hough = copy.deepcopy(black_and_white_image)
        standard_adaptive_hough = cv2.addWeighted(standard_adaptive_hough, 0.8, standard_adaptive_line_image, 1, 1)

        # build output image
        outputImage = get_image_grid([black_and_white_image, colourMasked,closed, image, edge, hough, standard_adaptive, standard_adaptive_hough] , row_length = 2)
        cv2.imshow("Lane Marking Detector", outputImage)

        return standard_adaptive_line_image