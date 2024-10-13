import cv2
import numpy as np
import copy
import feature.LaneDetection.LaneDetection as LaneDetection
import Capture.VideoFrameHandler as VideoFrameHandler

VideoFrameHandler = VideoFrameHandler.VideoFrameHandler('../data/Harard Warning Lights.mp4')
# get image
coloured = VideoFrameHandler.get_frame(0)

play_video = True

while(True):
    laneDetectionProcess = LaneDetection.LaneDetection.run_pipeline(coloured)
    cv2.imshow("Lane Marking Detector", laneDetectionProcess)

    if play_video:
        coloured = VideoFrameHandler.get_next_frame()

    k = cv2.waitKey(33)
    # print(k)
    if k==27: # Esc key
        break # stop
    elif k == 83: # right
        coloured = VideoFrameHandler.get_next_frame()
        pass # next image
    elif k == 81: # left
        coloured = VideoFrameHandler.get_previous_frame()
        pass
    elif k == 32: # space
        play_video = not play_video