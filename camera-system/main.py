from utils.CameraHarness import CameraHarness
from components.VideoWindow import VideoWindow

print("#"*10, "Camera System", "#"*10)

# Open Camera
webcam = CameraHarness(index=0, FPS=30)

window = VideoWindow(webcam)
window.showAndRecordFrames()
