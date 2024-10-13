import cv2


class EdgeDetectors:
    def applyCanny(image: cv2.typing.MatLike, LOWER_THRESHOLD = 150, UPPER_THRESHOLD = 290) -> cv2.typing.MatLike:
        return cv2.Canny(image, LOWER_THRESHOLD, UPPER_THRESHOLD)