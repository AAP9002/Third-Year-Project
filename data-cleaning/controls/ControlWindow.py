import tkinter as tk
from tkinter import ttk

class ControlWindow:
    def __init__(self, parent, title):
        self.parent = parent
        self.root = tk.Tk()
        self.root.title(title)
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def add_button(self, text, callback):
        button = ttk.Button(self.frame, text=text, command=callback)
        button.grid(sticky=(tk.W, tk.E))

    def add_slider(self, text, max_val, callback):
        label = ttk.Label(self.frame, text=text)
        label.grid(sticky=(tk.W, tk.E))
        slider = ttk.Scale(self.frame, from_=0, to=max_val, orient=tk.HORIZONTAL, command=lambda val: callback(int(float(val))))
        slider.grid(sticky=(tk.W, tk.E))

    def start(self):
        self.root.mainloop()