import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

cv2.namedWindow("SIFT", cv2.WINDOW_NORMAL)
cv2.namedWindow("SIFT Matches", cv2.WINDOW_NORMAL)
cv2.namedWindow("Vanishing Point", cv2.WINDOW_NORMAL)
cv2.namedWindow("X-axis Change Histogram", cv2.WINDOW_NORMAL)

class FeatureDetection():
    SHOW_BEST_FIT_GRAPH = True

    def __init__(self, image_width, image_height):
        self.feature_buffer_kp = []
        self.feature_buffer_des = []
        self.feature_buffer_size = 10
        self.image_width = image_width
        self.image_height = image_height


    def detect_features_by_SIFT(self, frame):
        frame_copy = copy.deepcopy(frame)

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
    
        list_paired_points = self.draw_feature_pairs(frame)
        if list_paired_points != None:
            predicted_vanishing_x_axis =  self.plot_x_differences(list_paired_points)
            print("Predicted vanishing point: ", predicted_vanishing_x_axis)
            frame_copy = cv2.line(frame_copy, (int(predicted_vanishing_x_axis), 0), (int(predicted_vanishing_x_axis), frame_copy.shape[0]), (0, 255, 0), 2)
            cv2.imshow("Vanishing Point", frame_copy)

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
            # # ignore out of center 50% of the image x
            # if kp1[match.queryIdx].pt[0] < frame.shape[1] * 0.25 or kp1[match.queryIdx].pt[0] > frame.shape[1] * 0.75:
            #     continue
            # # ignore out of center 50% of the image y
            # if kp1[match.queryIdx].pt[1] < frame.shape[0] * 0.25 or kp1[match.queryIdx].pt[1] > frame.shape[0] * 0.75:
            #     continue

            a = kp1[match.queryIdx]
            b = kp2[match.trainIdx]
            list_paired_points.append((a, b))

        # print("Paired points: ", list_paired_points[0][0].pt, list_paired_points[0][1].pt)

        return list_paired_points


    def plot_x_differences(self, paired_points):
        x_values = []
        x_differences = []
        
        for pt1, pt2 in paired_points:
            x1 = pt1.pt[0]  # x-coordinate of the first keypoint
            x2 = pt2.pt[0]  # x-coordinate of the second keypoint
            # remove outliers
            if abs(x1 - x2) > 50:
                continue

            if abs(x1 - x2) < 5:
                continue

            # center 70% of the image
            

            x_values.append(x1)
            x_differences.append(x1 - x2)

        # check neighbouring x values and mask out the outliers
        x_values = np.array(x_values)
        x_differences = np.array(x_differences)
        x_values = x_values[np.abs(x_differences - np.mean(x_differences)) < 2 * np.std(x_differences)]
        x_differences = x_differences[np.abs(x_differences - np.mean(x_differences)) < 2 * np.std(x_differences)]


        if len(x_values) == 0:
            return 0

        # line of best fit
        m, b = np.polyfit(x_values, x_differences, 1)

        # 2d curve fitting
        def func(x, a, b):
            return a*x + b
        
        popt, _ = curve_fit(func, x_values, x_differences)
        a, b = popt


        # Plotting the graph using matplotlib
        if self.SHOW_BEST_FIT_GRAPH:
            fig, ax = plt.subplots()
            ax.plot(x_values, x_differences, marker='o', linestyle='-', color='b')
            ax.set_title('X-coordinate Differences of Paired Points')
            ax.set_xlabel('X values on the frame')
            ax.set_ylabel('Difference in X')
            ax.set_xlim([0, self.image_width])
            ax.plot(x_values, m*np.array(x_values) + b, color='r')
            ax.grid(True)

            # Convert the plot to a NumPy array
            fig.canvas.draw()
            graph_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            graph_image = graph_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # Close the plot
            plt.close(fig)

            # Convert RGB to BGR for OpenCV
            graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)
            cv2.imshow("X-axis Change Histogram", graph_image)

        if m == 0:
            return 0

        # return where the line of best fit crosses the x-axis
        return -b//m

        