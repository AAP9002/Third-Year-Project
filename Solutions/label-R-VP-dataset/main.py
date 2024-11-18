### Open a video and label the frames with the VP dataset
from calendar import c
from time import sleep
import tkinter as tk
from tkinter import filedialog
import cv2

vp_positions = []
current_frame = 0

def getFileName():
    # open file dialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

def getCV2Video(filename):
    # open video file
    cap = cv2.VideoCapture(filename)
    return cap

def getFrameCount(cap):
    # get total number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_count

def getFrame(cap, index):
    # get frame at index
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    return frame

def getCursorPosition(event, x, y, flags, param):
    global current_pos
    global current_frame
    global vp_positions

    if event == cv2.EVENT_LBUTTONDOWN:
        current_frame += 10
        vp_positions.append((x, y))
        print(f'vp_positions: {vp_positions}')

def main():
    file_path = getFileName()
    print(f'file_path: {file_path}')
    cap = getCV2Video(file_path)
    frame_count = getFrameCount(cap)

    while current_frame < frame_count:
        frame = getFrame(cap, current_frame)
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('frame', getCursorPosition)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f'vp_positions: {vp_positions}')
    cv2.destroyAllWindows()
    cap.release()



main()