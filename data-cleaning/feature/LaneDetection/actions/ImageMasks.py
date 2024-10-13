import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageMasks:
    def maskImageAboveY(image:cv2.typing.MatLike, y_index:int):
        """mask all pixels above a given y value

        Args:
            image (MatLike): image passed by reference
            y_index (Integer): y value to mask at 
        """
        if(y_index > image.shape[0]):
            raise Exception(f'maskImageAboveY y_index out of rank for mask (y_index: {y_index}, image height: {image.shape[0]})')
        image[0:y_index, :] = 0

    def maskImageBelowY(image:cv2.typing.MatLike, y_index:int):
        """mask all pixels below a given y value

        Args:
            image (MatLike): image passed by reference
            y_index (Integer): y value to mask at 
        """
        if(y_index > image.shape[0]):
            raise Exception(f'maskImageAboveY y_index out of rank for mask (y_index: {y_index}, image height: {image.shape[0]})')
        image[y_index:, :] = 0

    def maskByColour(bgr_image:cv2.typing.MatLike, lower_range:int, upper_range:int) -> cv2.typing.MatLike:
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_range, upper_range)
        return mask
    
    def get_histogram_of_white_by_y_axis(BinaryImage):
        sum_y_axis = np.sum(BinaryImage, axis=1)
        average_of_white_per_row = sum_y_axis // BinaryImage.shape[1]
        return np.argmax(average_of_white_per_row < 150)

        