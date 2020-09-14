import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

class App:
    def __init__(self, master):
        # Constantes
        self.MODE_IMG = 0
        self.MODE_VIDEO = 1
        self.MODE_STREAM = 2
        self.IMG_WIDTH = 796
        self.IMG_HEIGHT = 597

        #Ventana principal
        self.window = master
        self.window.title = "MAIN WINDOW"

        # Menus

        self.MainMenu = tk.Menu(master)
        self.FileMenu = tk.Menu(self.MainMenu, tearoff=0)
        self.FileMenu.add_command(
                label="Open Image", command=self.filemenu_openi)
        self.FileMenu.add_command(
                label="Open Video", command=self.filemenu_openv)
        self.FileMenu.add_command(
                label="Open Stream", command=self.filemenu_opens)
        self.FileMenu.add_command(
                label="Save Annotations")
        self.FileMenu.add_command(label="Save Detected File")
        self.FileMenu.add_command(label="Quit", command=self.window.quit)
        self.MainMenu.add_cascade(label="File", menu=self.FileMenu)

        master.config(menu=self.MainMenu)

        # Frames
        self.FrameLeft = tk.Frame(master, relief=tk.RAISED, bg="black")
        self.FrameRight = tk.Frame(master)
        self.FrameLeft.pack(side=tk.LEFT, padx=5)
        self.FrameRight.pack(side=tk.LEFT)
        # Botones
        self.ButtonReproduce = tk.Button(self.FrameLeft, text="\uf04b") # 
        self.ButtonStop = tk.Button(self.FrameLeft, text="\uf04d") # 
        self.ButtonPause = tk.Button(self.FrameLeft, text="\uf04c") # 
        self.ButtonRecord = tk.Button(self.FrameLeft, text="\uf94a") # 壘
        
        self.ButtonReproduce.grid(row=0, column=0, padx=5, pady=5)
        self.ButtonPause.grid(row=0, column=1, padx=5, pady=5)
        self.ButtonStop.grid(row=0, column=2, padx=5, pady=5)
        self.ButtonRecord.grid(row=1, column=0, padx=5, pady=5)
        # Image Options
        # Video Options
        # Stream Options
        # Canvas

        self.CanvasMainImage = tk.Canvas(
                                    self.FrameRight, 
                                    width=self.IMG_WIDTH, 
                                    height=self.IMG_HEIGHT)
        self.CanvasMainImage.grid(
                row=0, column=0,
                padx=10, pady=10,
                sticky=tk.W + tk.S,
                )


        # Variables internas
        self.photo = None
        self.caption = None
        self.mode = None

        # Main
        self.show()
        self.window.mainloop()
        self.window.destroy()


    # Funciones generales 
    def show(self):
        fps = 25
        if self.mode == self.MODE_IMG:
            try:
                self.caption.release()
            except AttributeError:
                pass

            frame = cv2.imread(self.filepath)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(
                            frame, 
                            dsize=(self.IMG_WIDTH, self.IMG_HEIGHT), 
                            interpolation=cv2.INTER_AREA)

        elif self.mode == self.MODE_STREAM: 
            ret, frame = self.caption.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(
                            frame, 
                            dsize=(self.IMG_WIDTH, self.IMG_HEIGHT), 
                            interpolation=cv2.INTER_AREA)

        elif self.mode == self.MODE_VIDEO: 
            fps = self.caption.get(cv2.CAP_PROP_FPS)
            ret, frame = self.caption.read()
            if ret == True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(
                                frame, 
                                dsize=(self.IMG_WIDTH, self.IMG_HEIGHT), 
                                interpolation=cv2.INTER_AREA)
            else:
                self.mode = None

        else:
            ValueError(f"invalid self.mode value: {self.mode}")

        if self.mode == None:
            frame = np.zeros((self.IMG_HEIGHT,self.IMG_WIDTH,3), dtype=np.uint8)
            try:
                self.caption.release()
            except AttributeError:
                pass

        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.CanvasMainImage.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(int(1000 * (1/fps)), self.show)

    
    # Funciones widgets 

    # Menus

    def filemenu_openi(self):
        """Open Image"""
        self.filepath = filedialog.askopenfilename(
                    initialdir="./",
                    title="Select File",
                    filetypes = (
                        ("jpg files", "*.jpg"), 
                        ("png files", "*.png"),))

        self.mode = self.MODE_IMG

    def filemenu_openv(self):
        """Open Video"""
        self.filepath = filedialog.askopenfilename(
                    initialdir="./",
                    title="Select File", 
                    filetypes = (
                        ("avi files", "*.avi"),
                        ("mp4 files", "*.mp4"),))


        self.mode = self.MODE_VIDEO
        self.caption = cv2.VideoCapture(self.filepath)

    def filemenu_opens(self):
        """Open Video"""
        self.mode = self.MODE_STREAM
        self.caption = cv2.VideoCapture(0)

    # Botones



if __name__=="__main__":
    root = tk.Tk()
    app = App(root)
    #root.mainloop()
    #root.destroy()
