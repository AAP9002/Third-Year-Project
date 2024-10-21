import copy
from turtle import width
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from .utils.ImageStacker import get_image_grid
from .utils.VideoRecorder import VideoRecorder

cv2.namedWindow("SIFT", cv2.WINDOW_NORMAL)

class FeatureDetection():
    SHOW_BEST_FIT_GRAPH = True

    def __init__(self, image_width, image_height):
        self.feature_buffer_kp = []
        self.feature_buffer_des = []
        self.feature_buffer_size = 10
        self.image_width = image_width
        self.image_height = image_height
        self.output_image_step = []
        self.video_recorder = VideoRecorder("feature_detection_output.avi", 10)


    def detect_features_by_SIFT(self, frame):
        frame_copy = copy.deepcopy(frame)

        self.output_image_step = []

        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(frame, None)
        frame = cv2.drawKeypoints(frame, kp, frame, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        self.output_image_step.append(frame)
    
        if len(self.feature_buffer_kp) > self.feature_buffer_size:
            self.feature_buffer_des.pop(0)
            self.feature_buffer_kp.pop(0)
        self.feature_buffer_des.append(des)
        self.feature_buffer_kp.append(kp)

        print("Number of features: ", len(kp))
        print("Number of features in buffer: ", len(self.feature_buffer_kp))
        print("Number of feature pairs: ", len(self.get_feature_pairs()))
    
        list_paired_points = self.draw_feature_pairs(frame)
        if list_paired_points != None:
            predicted_vanishing_x_axis =  self.plot_x_differences(list_paired_points)
            print("Predicted vanishing point: ", predicted_vanishing_x_axis)
            cv2.line(frame_copy, (int(predicted_vanishing_x_axis), 0), (int(predicted_vanishing_x_axis), frame_copy.shape[0]-2), (0, 255, 0), 3)

            predicted_vanishing_y_axis =  self.plot_y_differences(list_paired_points)
            print("Predicted vanishing point: ", predicted_vanishing_y_axis)
            cv2.line(frame_copy, (0, int(predicted_vanishing_y_axis)), (frame_copy.shape[1]-2, int(predicted_vanishing_y_axis)), (0, 255, 0), 3)

            self.output_image_step.append(frame_copy)

        # reorder the output image
        if len(self.output_image_step) < 5:
            return frame
        
        image_layout = [self.output_image_step[0], self.output_image_step[5], self.output_image_step[4], self.output_image_step[1], self.output_image_step[2], self.output_image_step[3]]
        
        output_image = get_image_grid(image_layout, row_length=3)
        self.video_recorder.record_frame(output_image)
        cv2.imshow("SIFT", output_image)

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
        # split into 2 images
        self.output_image_step.append(np.hsplit(frame, 2)[0])
        self.output_image_step.append(np.hsplit(frame, 2)[1])
        # cv2.imshow("SIFT Matches", frame)

        list_paired_points = []
        for match in matches:
            a = kp1[match.queryIdx]
            b = kp2[match.trainIdx]
            list_paired_points.append((a, b))

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
            if x1 < 0.15 * self.image_width or x1 > 0.85 * self.image_width:
                continue

            x_values.append(x1)
            x_differences.append(x1 - x2)

        # filter for points with 5 closest neighbours
        data = np.column_stack((x_values, x_differences))
        if len(data) < 15:
            return 0
        neigh = NearestNeighbors(n_neighbors=15)
        neigh.fit(data)
        distances, indices = neigh.kneighbors(data)
        indices = indices[distances[:, 4] < 30]
        x_values = [x_values[i] for i in indices.flatten()]
        x_differences = [x_differences[i] for i in indices.flatten()]

        # check neighbouring x values and mask out the outliers
        x_values = np.array(x_values)
        x_differences = np.array(x_differences)
        x_values = x_values[np.abs(x_differences - np.mean(x_differences)) < 2 * np.std(x_differences)]
        x_differences = x_differences[np.abs(x_differences - np.mean(x_differences)) < 2 * np.std(x_differences)]


        if len(x_values) == 0:
            return 0

        # line of best fit
        m, b = np.polyfit(x_values, x_differences, 1)

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


            # Convert the plot to a NumPy array with width = self.image_width and height = self.image_height
            fig.canvas.draw()
            graph_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            graph_image = graph_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # Close the plot
            plt.close(fig)

            # Convert RGB to BGR for OpenCV
            graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)
            graph_image = cv2.resize(graph_image, (self.image_width, self.image_height))
            # cv2.imshow("X-axis Change Histogram", graph_image)
            self.output_image_step.append(graph_image)

        if m == 0:
            return 0

        # return where the line of best fit crosses the x-axis
        return -b//m
    
    def plot_y_differences(self, paired_points):
        y_values = []
        y_differences = []
        
        for pt1, pt2 in paired_points:
            y1 = pt1.pt[1]
            y2 = pt2.pt[1]
            # remove outliers
            if abs(y1 - y2) > 50:
                continue

            if abs(y1 - y2) < 3:
                continue

            # center 50% of the image
            if y1 < 0.25 * self.image_height or y1 > 0.60 * self.image_height:
                continue

            y_values.append(y1)
            y_differences.append(y1 - y2)

        # filter for points with 5 closest neighbours
        data = np.column_stack((y_values, y_differences))
        if len(data) < 15:
            return 0
        neigh = NearestNeighbors(n_neighbors=15)
        neigh.fit(data)
        distances, indices = neigh.kneighbors(data)
        indices = indices[distances[:, 4] < 30]
        y_values = [y_values[i] for i in indices.flatten()]
        y_differences = [y_differences[i] for i in indices.flatten()]

        # check neighbouring x values and mask out the outliers
        y_values = np.array(y_values)
        y_differences = np.array(y_differences)
        y_values = y_values[np.abs(y_differences - np.mean(y_differences)) < 2 * np.std(y_differences)]
        y_differences = y_differences[np.abs(y_differences - np.mean(y_differences)) < 2 * np.std(y_differences)]

        if len(y_values) == 0:
            return 0

        # line of best fit
        m, b = np.polyfit(y_values, y_differences, 1)

        # Plotting the graph using matplotlib

        fig, ax = plt.subplots()
        ax.plot(y_values, y_differences, marker='o', linestyle='-', color='b')
        ax.set_title('Y-coordinate Differences of Paired Points')
        ax.set_xlabel('Y values on the frame')
        ax.set_ylabel('Difference in Y')
        ax.set_xlim([0, self.image_height])
        ax.plot(y_values, m*np.array(y_values) + b, color='r')
        ax.grid(True)

        # Convert the plot to a NumPy array
        fig.canvas.draw()
        graph_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        graph_image = graph_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        # Convert RGB to BGR for OpenCV
        graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)
        graph_image = cv2.resize(graph_image, (self.image_width, self.image_height))
        # cv2.imshow("Y-axis Change Histogram", graph_image)
        self.output_image_step.append(graph_image)

        if m == 0:
            return 0
        
        # return where the line of best fit crosses the x-axis
        return -b//m

        