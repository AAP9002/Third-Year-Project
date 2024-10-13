import cv2

MAX_THRESHOLD_VALUE = 255


class ThresholdMethods:
    def adaptiveThesholding(image:cv2.typing.MatLike, BLOCK_SIZE = 15, CONSTANT = 5):
        """apply adaptive thesholding to image

        Args:
            image (cv2.typing.MatLike): image passed by reference
            BLOCK_SIZE (int, optional): dimensions to consider as local neighbours. Defaults to 15.
            CONSTANT (int, optional): adjust sensitivity of decision boundary. Defaults to 5.
        """
        cv2.adaptiveThreshold(
                image,
                MAX_THRESHOLD_VALUE, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY,
                BLOCK_SIZE,
                CONSTANT,
                image
            )