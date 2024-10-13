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
    
    def get_sky_y_axis(Image):
        _, OTSU_threshold_image = cv2.threshold(Image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(OTSU_threshold_image)
        sum_y_axis = np.sum(OTSU_threshold_image, axis=1)
        average_of_white_per_row = sum_y_axis // OTSU_threshold_image.shape[1]

        # reverse to search bottom up
        reversed_avg = average_of_white_per_row[::-1]
        
        # Find the first row where the white pixel count drops significantly
        bottom_up_index = np.argmax(reversed_avg > 130)
        
        # convert back to top down
        return len(average_of_white_per_row) - bottom_up_index - 1
        