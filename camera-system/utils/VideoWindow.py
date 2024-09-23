import cv2
from .FrameFileHandler import FrameFileHandler

class VideoWindow:
    def __init__(self, cameraInstance, savePath='output.mp4') -> None:
        self.cameraInstance = cameraInstance
        self.frameFIleHandler = FrameFileHandler(cameraInstance, savePath)

    def showAndRecordFrames(self):
        while True:
            frame = self.cameraInstance.getNextFrame()
            
            # print(frame)
            self.frameFIleHandler.saveFrameToFile(frame)

            # Display frame
            cv2.imshow('Camera', frame)

            if cv2.waitKey(1) == ord('q'):
                break

    def __del__(self):
        cv2.destroyWindow('Camera')