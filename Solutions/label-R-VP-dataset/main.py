### Open a video and label the frames with the VP dataset
import tkinter as tk
from tkinter import filedialog

def getFileName():
    # open file dialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

def main():
    file_path = getFileName()
    print(f'file_path: {file_path}')

main()