import cv2
import numpy as np
from .FrameFileHandler import FrameFileHandler

class VideoWindow:
    def __init__(self, cameraInstance, savePath='output.mp4') -> None:
        self.cameraInstance = cameraInstance
        self.frameFIleHandler = FrameFileHandler(cameraInstance.frame_width*2, cameraInstance.frame_height*2, cameraInstance.FPS, savePath)

    def showAndRecordFrames(self):
        while True:
            frame_1 = self.cameraInstance.getNextFrame()
            
            frame = combineImagesIntoQuadrant(frame_1, frame_1, frame_1, frame_1)

            # print(frame)
            self.frameFIleHandler.saveFrameToFile(frame)

            # Display frame
            cv2.imshow('Camera', frame)

            if cv2.waitKey(1) == ord('q'):
                break

    def __del__(self):
        cv2.destroyWindow('Camera')

def combineImagesIntoQuadrant(frameNW, frameNE, frameSW, frameSE):
    top_row = np.hstack((frameNW, frameNE))
    bottom_row = np.hstack((frameSW, frameSE))
    combined_frame = np.vstack((top_row, bottom_row))
    return combined_frame