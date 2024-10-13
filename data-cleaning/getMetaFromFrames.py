import cv2
import Capture.VideoFrameHandler as VideoFrameHandler
import pytesseract

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Cropped", cv2.WINDOW_NORMAL)

VideoFrameHandler = VideoFrameHandler.VideoFrameHandler('../data/20241004_182941000_iOS.mp4')
# get image
coloured = VideoFrameHandler.get_frame(0)

# only accept numbers
custom_config = r'--oem 3 --psm 6 outputbase digits'

while(True):
    coloured = VideoFrameHandler.get_next_frame()

    # crop to top left corner
    cropped = coloured[80:125, 0:440]
    mph_crop = coloured[80:125, 400:440]
    # select white pixels
    white = cv2.inRange(cropped, (210, 210, 210), (255, 255, 255))
    # white = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # invert colour
    white = cv2.bitwise_not(white)
    # smooth
    cv2.medianBlur(white, 5)
    print(pytesseract.image_to_string(cropped, config=custom_config))
    cv2.imshow("Original", coloured)
    cv2.imshow("Cropped", white)

    k = cv2.waitKey(33)
    # print(k)
    if k==27: # Esc key
        break # stop