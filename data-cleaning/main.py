from multiprocessing import process
import cv2
import numpy as np
import copy
import feature.LaneDetection.LaneDetection as LaneDetection
import feature.FeatureDetection.FeatureDetection as FeatureDetection
import Capture.VideoFrameHandler as VideoFrameHandler


# VideoFrameHandler = VideoFrameHandler.VideoFrameHandler('../data/Know you-re protected 1080p webloop.mp4')
# VideoFrameHandler = VideoFrameHandler.VideoFrameHandler('../data/10 sec video 1 mototrway crash h.264.mp4')
# VideoFrameHandler = VideoFrameHandler.VideoFrameHandler('../data/Harard Warning Lights.mp4')
VideoFrameHandler = VideoFrameHandler.VideoFrameHandler('../data/20241004_182941000_iOS.mp4')
# VideoFrameHandler = VideoFrameHandler.VideoFrameHandler('../data/2022_0813_184754_009.MP4')

def set_frame(frame_number:int):
    global coloured
    coloured = VideoFrameHandler.get_frame(frame_number)

# get image
coloured = VideoFrameHandler.get_frame(0)

featureDetectionHandler = FeatureDetection.FeatureDetection(image_width=coloured.shape[1], image_height=coloured.shape[0])

play_video = True

def detect_lanes():
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    processed_image = copy.deepcopy(coloured)
    laneDetectionLine = LaneDetection.LaneDetection.run_pipeline(copy.deepcopy(coloured))
    black_white_to_red = cv2.cvtColor(laneDetectionLine, cv2.COLOR_GRAY2BGR)
    black_white_to_red[:, :, 1] = 0
    processed_image = cv2.addWeighted(processed_image, 0.8, black_white_to_red, 1, 1)
    cv2.imshow("Original", processed_image)


while(True):
    # detect_lanes()

    # apply sift
    sift_image = copy.deepcopy(coloured)
    featureDetectionHandler.detect_features_by_SIFT(sift_image)

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