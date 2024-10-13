import cv2
import numpy as np
import copy
import feature.LaneDetection.LaneDetection as LaneDetection
import Capture.VideoFrameHandler as VideoFrameHandler

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)

# VideoFrameHandler = VideoFrameHandler.VideoFrameHandler('../data/Harard Warning Lights.mp4')
# VideoFrameHandler = VideoFrameHandler.VideoFrameHandler('../data/10 sec video 1 mototrway crash h.264.mp4')
VideoFrameHandler = VideoFrameHandler.VideoFrameHandler('../data/20241004_182941000_iOS.mp4')


def set_frame(frame_number:int):
    global coloured
    coloured = VideoFrameHandler.get_frame(frame_number)

# get image
coloured = VideoFrameHandler.get_frame(0)

play_video = True

while(True):
    cv2.imshow("Original", coloured)

    laneDetectionProcess = LaneDetection.LaneDetection.run_pipeline(copy.deepcopy(coloured))
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