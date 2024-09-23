import cv2

class VideoWindow:
    def __init__(self, cameraInstance, savePath='output.mp4') -> None:
        self.cameraInstance = cameraInstance
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            savePath,
            fourcc,
            cameraInstance.FPS, 
            (cameraInstance.frame_width, cameraInstance.frame_height)
            )

    def showAndRecordFrames(self):
        while True:
            frame = self.cameraInstance.getNextFrame()
            # print(frame)

            # Write frame to file
            self.out.write(frame)

            # Display frame
            cv2.imshow('Camera', frame)

            if cv2.waitKey(1) == ord('q'):
                break

    def __del__(self):
        self.out.release()
        cv2.destroyWindow('Camera')