import cv2
import numpy as np
import copy
import feature.LaneDetection.LaneDetection as LaneDetection
import feature.FeatureDetection.FeatureDetection as FeatureDetection
import Capture.VideoFrameHandler as VideoFrameHandler


# VideoFrameHandler = VideoFrameHandler.VideoFrameHandler('../data/Know you-re protected 1080p webloop.mp4')
# VideoFrameHandler = VideoFrameHandler.VideoFrameHandler('../data/10 sec video 1 mototrway crash h.264.mp4')
# VideoFrameHandler = VideoFrameHandler.VideoFrameHandler('../data/Harard Warning Lights.mp4')
# VideoFrameHandler = VideoFrameHandler.VideoFrameHandler('../data/20241004_182941000_iOS.mp4')
VideoFrameHandler = VideoFrameHandler.VideoFrameHandler('../data/2022_0813_184754_009.MP4')

def set_frame(coloured_in:cv2.typing.MatLike):
    global coloured
    coloured = coloured_in

    processed_image = copy.deepcopy(coloured)

    lanes_overlay = detect_lanes()
    processed_image = cv2.addWeighted(processed_image, 0.8, lanes_overlay, 1, 1)
    
    detect_features()

    cv2.imshow("Original", processed_image)

# get image
coloured = VideoFrameHandler.get_frame(0)

featureDetectionHandler = FeatureDetection.FeatureDetection(image_width=coloured.shape[1], image_height=coloured.shape[0])

play_video = True

def detect_lanes():
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    laneDetectionLine = LaneDetection.LaneDetection.run_pipeline(copy.deepcopy(coloured))
    black_white_to_red = cv2.cvtColor(laneDetectionLine, cv2.COLOR_GRAY2BGR)
    black_white_to_red[:, :, 1] = 0
    return black_white_to_red

def detect_features():
    sift_image = copy.deepcopy(coloured)
    featureDetectionHandler.detect_features_by_SIFT(sift_image)


while(True):
    if play_video:
        # apply sift
        set_frame(VideoFrameHandler.get_next_frame())

    k = cv2.waitKey(33)
    # print(k)
    if k==27: # Esc key
        break # stop
    elif k == 83: # right
        set_frame(VideoFrameHandler.get_next_frame())

        pass # next image
    elif k == 81: # left
        set_frame(VideoFrameHandler.get_previous_frame())
        pass
    elif k == 32: # space
        play_video = not play_video