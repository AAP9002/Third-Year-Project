import cv2

class FeatureDetection():

    def detect_features_by_SIFT(self, frame):
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(frame, None)
        frame = cv2.drawKeypoints(frame, kp, frame, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        cv2.imshow("SIFT", frame)
        
        return frame