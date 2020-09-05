import cv2
import tkinter as tk
from tkinter import W, E, S, N, filedialog
from PIL import Image, ImageTk

class App:
    def __init__(self, master):
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
        
        # Botones
        # Image Options
        # Video Options
        self.Button_probe = tk.Button(master, text="Prueba")
        self.Button_probe.grid(row=0, column=0, padx=5, pady=5)
        # Stream Options
        # Canvas

        self.CanvasMainImage = tk.Canvas(master, width=796, height=591)
        self.CanvasMainImage.grid(
                row=0, column=1,
                padx=10, pady=10,
                sticky=W+S,
                )

        # Constantes
        self.MODE_IMG = 0
        self.MODE_VIDEO = 1
        self.MODE_STREAM = 2
        self.IMG_WIDHT = 796
        self.IMG_HEIGHT = 597

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
        if self.mode == self.MODE_IMG:
            img = cv2.imread(self.filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(
                    img, dsize=(self.IMG_WIDHT, self.IMG_WIDHT), 
                    fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.CanvasMainImage.create_image(0, 0, image=self.photo, anchor=tk.NW)

        elif self.mode == self.MODE_STREAM or self.mode == self.MODE_VIDEO:
            ret, frame = self.caption.read()
            frame = cv2.resize(
                    frame, dsize=(self.IMG_WIDHT, self.IMG_WIDHT), 
                    fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.CanvasMainImage.create_image(0, 0, image=self.photo, anchor=tk.NW)

        elif self.mode == None:
            pass
        
        else:
            ValueError(f"invalid self.mode value: {self.mode}")

        self.window.after(40, self.show)

    
    # Funciones widgets 

    # Menus

    def filemenu_openi(self):
        """Open Image"""
        self.filepath = filedialog.askopenfilename(
                    initialdir="./",
                    title="Select File",)

        self.mode = self.MODE_IMG

    def filemenu_openv(self):
        """Open Video"""
        self.filepath = filedialog.askopenfilename(
                    initialdir="./",
                    title="Select File")

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
