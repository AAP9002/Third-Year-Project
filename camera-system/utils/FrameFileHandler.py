import cv2

class FrameFileHandler:
    def __init__(self, cameraInstance, savePath='output.mp4' ):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            savePath,
            fourcc,
            cameraInstance.FPS, 
            (cameraInstance.frame_width, cameraInstance.frame_height)
            )
        self.current_frame_number = 0

    def saveFrameToFile(self, frame):
        self.out.write(frame)

    def __del__(self):
        self.out.release()