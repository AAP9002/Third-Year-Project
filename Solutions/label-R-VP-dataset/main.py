### Open a video and label the frames with the VP dataset
import tkinter as tk
from tkinter import filedialog
import cv2

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

def main():
    file_path = getFileName()
    print(f'file_path: {file_path}')
    cap = getCV2Video(file_path)
    frame_count = getFrameCount(cap)
    
    for i in range(frame_count):
        frame = getFrame(cap, i)
        cv2.imshow('frame', frame)
        
    
    cap.release()
    cv2.destroyAllWindows()


main()