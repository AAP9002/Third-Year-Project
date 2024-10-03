import cv2
import numpy as np

# get image
originalImage = cv2.imread('./lane_detection_7.jpeg', cv2.IMREAD_GRAYSCALE)
# originalImage = cv2.imread('./lane_detection_5.jpeg')
# originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
image = originalImage

# smoothing
image = cv2.GaussianBlur(image,(5,5),0)

# threshold for white
(T, image) = cv2.threshold(image, 200, 255, cv2.THRESH_OTSU)
print('OTSU Threshold:', T)

# canny edge detection
t_lower = 50
t_upper = 150
  
# Applying the Canny Edge filter 
edge = cv2.Canny(image, t_lower, t_upper) 

# build output image
outputImage = np.hstack((originalImage, image))
outputImage = np.hstack((outputImage, edge))

cv2.namedWindow("hi", cv2.WINDOW_NORMAL) 
cv2.imshow('hi', outputImage)

cv2.waitKey(0)