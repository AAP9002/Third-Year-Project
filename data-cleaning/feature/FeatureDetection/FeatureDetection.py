from operator import le
import cv2
from numpy import imag

cv2.namedWindow("SIFT", cv2.WINDOW_NORMAL)
cv2.namedWindow("SIFT Matches", cv2.WINDOW_NORMAL)

class FeatureDetection():
    def __init__(self):
        self.feature_buffer_kp = []
        self.feature_buffer_des = []
        self.feature_buffer_size = 10

    def detect_features_by_SIFT(self, frame):
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(frame, None)
        frame = cv2.drawKeypoints(frame, kp, frame, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
        if len(self.feature_buffer_kp) > self.feature_buffer_size:
            self.feature_buffer_des.pop(0)
            self.feature_buffer_kp.pop(0)
        self.feature_buffer_des.append(des)
        self.feature_buffer_kp.append(kp)

        cv2.imshow("SIFT", frame)

        print("Number of features: ", len(kp))
        print("Number of features in buffer: ", len(self.feature_buffer_kp))
        print("Number of feature pairs: ", len(self.get_feature_pairs()))
    
        self.draw_feature_pairs(frame)
        return frame
    
    def get_feature_pairs(self):
        if len(self.feature_buffer_des) < 2:
            return []
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.feature_buffer_des[-1], self.feature_buffer_des[-2], k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        return good

    def draw_feature_pairs(self, frame):
        if len(self.feature_buffer_des) < 2:
            return None
        kp1 = cv2.KeyPoint_convert([kp.pt for kp in self.feature_buffer_kp[-1]])
        kp2 = cv2.KeyPoint_convert([kp.pt for kp in self.feature_buffer_kp[-2]])
        matches = self.get_feature_pairs()
        frame = cv2.drawMatches(frame, kp1, frame, kp2, matches, frame)
        cv2.imshow("SIFT Matches", frame)

        list_paired_points = []
        for match in matches:
            list_paired_points.append((kp1[match.queryIdx], kp2[match.trainIdx]))

        return list_paired_points