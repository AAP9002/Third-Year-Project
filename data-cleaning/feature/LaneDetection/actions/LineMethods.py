import cv2
import numpy as np


class LineMethods:
    def applyHoughLines(image: cv2.typing.MatLike, threshold:int=50, minLineLength:int=50, maxLineGap:int=20):
        lines = cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=threshold, 
                            minLineLength=minLineLength, maxLineGap=maxLineGap)

        line_image = np.zeros_like(image)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (155, 0, 0), 5)
                    

        return line_image