import cv2

class VideoFrameHandler:
    currentFrame = 0

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.totalFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_increment = self.fps // 10

    def get_frame(self, frame_number:int):
        print("Getting frame: ", frame_number, " of ", self.totalFrames)
        self.currentFrame = frame_number
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame
    
    def get_next_frame(self):
        self.currentFrame += self.frame_increment
        return self.get_frame(self.currentFrame)
    
    def get_previous_frame(self):
        self.currentFrame -= self.frame_increment
        return self.get_frame(self.currentFrame)
    
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()