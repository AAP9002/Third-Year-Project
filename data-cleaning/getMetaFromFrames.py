import cv2
import Capture.VideoFrameHandler as VideoFrameHandler
import pytesseract
from feature.LaneDetection.actions.SmoothingMethods import SmoothingMethods

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Cropped", cv2.WINDOW_NORMAL)

# VideoFrameHandler = VideoFrameHandler.VideoFrameHandler('../data/Know you-re protected 1080p webloop.mp4')
VideoFrameHandler = VideoFrameHandler.VideoFrameHandler('../data/10 sec sheep h.264.mp4')
# get image
coloured = VideoFrameHandler.get_frame(0)

# only accept numbers
custom_config = r'--oem 3 --psm 6 outputbase digits'

while(True):
    coloured = VideoFrameHandler.get_next_frame()

    # print(coloured.shape)

    # crop to top left corner
    # cropped = coloured[800:100, 1000:1900]
    cropped = coloured[-45:, :]

    # apply closing
    # SmoothingMethods.applyClosing(cropped)

    # select white pixels
    # white = cv2.inRange(cropped, (210, 210, 210), (255, 255, 255))
    # white = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # invert colour
    # white = cv2.bitwise_not(white)
    # smooth
    # cv2.medianBlur(white, 5)
    # print(pytesseract.image_to_string(cropped, config=custom_config))
    print(pytesseract.image_to_string(cropped))
    cv2.imshow("Original", coloured)
    cv2.imshow("Cropped", cropped)

    k = cv2.waitKey(33)
    # print(k)
    if k==27: # Esc key
        break # stop