import cv2

class VideoRecorder():
    def __init__(self, fileName:str, fps:int) -> None:
        self.fileName = fileName
        self.fps = fps
        self.video = None

    def record_frame(self, frame):
        if self.video is None:
            self.video = cv2.VideoWriter(self.fileName, cv2.VideoWriter_fourcc(*'XVID'), self.fps, (frame.shape[1], frame.shape[0]))
        print("Recording frame")
        self.video.write(frame)
    
    def release(self):
        self.video.release()

    def __del__(self):
        self.release()
        print("Video Recorder released")