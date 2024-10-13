import cv2
import numpy as np
import copy
import feature.LaneDetection.LaneDetection as LaneDetection

# get image
coloured = cv2.imread('../data/lane_detection_7.jpeg')

while(True):
    laneDetectionProcess = LaneDetection.LaneDetection.run_pipeline(coloured)
    cv2.imshow("Lane Marking Detector", laneDetectionProcess)

    k = cv2.waitKey(33)
    # print(k)
    if k==27: # Esc key
        break # stop
    elif k == 83: # right
        print("right arrow pressed")
        pass # next image