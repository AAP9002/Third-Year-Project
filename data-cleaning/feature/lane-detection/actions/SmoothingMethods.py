import cv2


class SmoothingMethods:
    def GaussianBlur(image:cv2.typing.MatLike, GAUSSIAN_BLUR_KERNEL_SIZE:tuple[int, int]=(9,9)):
        """apply GaussianBlur to image

        Args:
            image (cv2.typing.MatLike): image passed by reference
            GAUSSIAN_BLUR_KERNEL_SIZE (tuple[int, int], optional): kernel size to apply on image. Defaults to (9,9).
        """
        cv2.GaussianBlur(image,GAUSSIAN_BLUR_KERNEL_SIZE,0,image)