import cv2

class FrameFileHandler:
    def __init__(self, target_width, target_height, FPS, savePath='output.mp4' ):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            savePath,
            fourcc,
            FPS, 
            (target_width, target_height)
            )
        self.current_frame_number = 0

    def saveFrameToFile(self, frame):
        self.out.write(frame)

    def __del__(self):
        self.out.release()