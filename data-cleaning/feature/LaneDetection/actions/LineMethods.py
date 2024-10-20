import cv2
import numpy as np

def angle_from_north(x1, y1, x2, y2):
    return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi


class LineMethods:
    def applyHoughLines(image: cv2.typing.MatLike, threshold:int=50, minLineLength:int=50, maxLineGap:int=20) -> cv2.typing.MatLike:
        lines = cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=threshold, 
                            minLineLength=minLineLength, maxLineGap=maxLineGap)

        line_image = np.zeros_like(image)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # filter for vertical lines
                if abs(angle_from_north(x1, y1, x2, y2)) > 23:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 5)
                    

        return line_image