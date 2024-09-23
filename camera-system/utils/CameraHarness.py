import cv2

class CameraHarness:
    def __init__(self, index, FPS) -> None:
        self.index = index
        self.FPS = FPS
        self.cam = cv2.VideoCapture(index)
        self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def getNextFrame(self):
        ret, frame = self.cam.read()
        return frame

    # destructor to release camera resource
    def __del__(self):
        self.cam.release()
        print(f"CameraHarness - index {self.index} released and destroyed")