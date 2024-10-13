import cv2
import numpy as np


class SmoothingMethods:
    def GaussianBlur(image:cv2.typing.MatLike, GAUSSIAN_BLUR_KERNEL_SIZE:tuple[int, int]=(9,9)):
        """apply GaussianBlur to image

        Args:
            image (cv2.typing.MatLike): image passed by reference
            GAUSSIAN_BLUR_KERNEL_SIZE (tuple[int, int], optional): kernel size to apply on image. Defaults to (9,9).
        """
        cv2.GaussianBlur(image,GAUSSIAN_BLUR_KERNEL_SIZE,0,image)

    def applyClosing(image:cv2.typing.MatLike):
        kernel = np.ones((5, 5), np.uint8)
        cv2.dilate(image, kernel, image)
        cv2.erode(image, kernel, image)