import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from classDeepSort import DeepSort
from yolov3_tf2.utils import draw_YOLO, draw_DS
import os 

"""
Cosas por hacer:

-> actualizar la funciones draw_YOLO para que muestre clase y score (listo)

-> agregar boton de stop para cerrar los archivos a guardar 



"""


class App:
    def __init__(self, master):
        # Constantes
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
        self.ButtonReproduce = tk.Button(self.FrameLeft, 
                                         command=self.button_reproduce,
                                         text="PLAY") # 
        #self.ButtonStop = tk.Button(self.FrameLeft, 
        #                            command=self.button_stop,
        #                            text="STOP") # 
        self.ButtonPause = tk.Button(self.FrameLeft, 
                                     command=self.button_pause,
                                     text="PAUSE") # 
        #self.ButtonRecord = tk.Button(self.FrameLeft, text="\uf94a") # 壘
        
        self.checkboxDec = tk.Checkbutton(self.FrameLeft, text="Detector",command= self.toggleDetector)
        self.checkboxDec.grid(row=0, column=0, padx=5, pady=5)
        self.checkboxDec.select()
        self.checkboxTra = tk.Checkbutton(self.FrameLeft, text="Tracker",command= self.toggleTracker)
        self.checkboxTra.grid(row=0, column=1, padx=5, pady=5)
        self.checkboxTra.select()        
        
        self.ButtonReproduce.grid(row=1, column=0, padx=5, pady=5)
        self.ButtonPause.grid(row=1, column=1, padx=5, pady=5)
        
        #self.ButtonStop.grid(row=0, column=2, padx=5, pady=5)
        #self.ButtonRecord.grid(row=1, column=0, padx=5, pady=5)
        # Image Options
        # Video Options
        self.MODE_VIDEO_REPRODUCE = True  

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
        self.frame = None
        self.photo = None
        self.caption = None
        self.fps = 25
        self.filepath = os.path.join('data','girl.png')
        
        
        #variable de tracker 
        self.tracker = DeepSort() 
        self.typeTracker = True # True si se quiere usar como tracker
        self.typeDetector = True #True si se quiere usar como detector           
        
        #variables para guardar 
        self.saveAnnotations = None 
        self.saveVideo = None 
        
        #variables mode 
        self.modeFunction = self.modeimage #por default 
        

        # Main
        self.update()
        self.window.mainloop()
        
        #all the things tha you need to close put here 
        
        # Solución problema de cierre de ventana 
        try:
            self.window.destroy()
        except:
            pass

    def update(self):
        """Función para actualizar en screen
        poner todo lo que se requiera actualizar"""   
        
        
        # modeFunction: imagen, video o streaming
        # actializa frame cargando desde la fuente  
        if self.MODE_VIDEO_REPRODUCE:
            self.modeFunction()
            self.trackdetec()

           
        self.photo = cv2.resize(self.frame, 
                                dsize=(self.IMG_WIDTH, self.IMG_HEIGHT), 
                                interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.photo))
        self.CanvasMainImage.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(int(1000 * (1/self.fps)), self.update)
        
        
    def trackdetec(self):
        """Actualiza frame con detecciones o tracks """
        if self.typeDetector or self.typeTracker: 
            boxes_ds, id_ds ,boxes_nms, sco_nms, classIDs_nms, ids_nms, scales_nms, class_names  = self.tracker(self.frame)
        if self.typeTracker:    
            self.frame = draw_DS(self.frame, boxes_ds, id_ds)    # para pintar el traker 
        if self.typeDetector:
            self.frame = draw_YOLO(self.frame, (boxes_nms, sco_nms, classIDs_nms, ids_nms, # para pintar el detector 
                              scales_nms), class_names)
    	

    def modeimage(self):
        """Actualiza frame con imagen cargada"""
        try:
            self.caption.release()
        except AttributeError:
            pass
        self.frame = cv2.imread(self.filepath)
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        
    def modevideo(self):
        self.fps = self.caption.get(cv2.CAP_PROP_FPS)
        ret, frame = self.video_handler()
        if ret == True:
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            self.MODE_VIDEO_REPRODUCE = False 
        	
        

    def modestream(self):
        ret, self.frame = self.caption.read()
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
    
                  
    def video_handler(self):
        if self.MODE_VIDEO_REPRODUCE or self.photo == None:
            return self.caption.read()
        else:
            return True, cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)


    # Funciones widgets 

    # Menus

    def filemenu_openi(self):
        """Open Image"""
        filepath = filedialog.askopenfilename(
                    initialdir="./",
                    title="Select File",
                    filetypes = (
                        ("jpg files", "*.jpg"), 
                        ("png files", "*.png"),))
                        
        if filepath[-3:] in ["jpg", "png"]:
            self.MODE_VIDEO_REPRODUCE = True
            self.filepath = filepath
            self.modeFunction =  self.modeimage
            self.tracker.deepsort.reset_tracker()
        

    def filemenu_openv(self):
        """Open Video"""
        filepath = filedialog.askopenfilename(
                    initialdir="./",
                    title="Select File", 
                    filetypes = (
                        ("avi files", "*.avi"),
                        ("mp4 files", "*.mp4"),))
                        
        if filepath[-3:] in ["avi", "mp4"]:
            self.MODE_VIDEO_REPRODUCE = True
            self.filepath = filepath
            self.caption = cv2.VideoCapture(self.filepath)
            self.modeFunction =  self.modevideo
            self.tracker.deepsort.reset_tracker()

    def filemenu_opens(self):
        """Open Stream"""
        self.MODE_VIDEO_REPRODUCE = True
        self.caption = cv2.VideoCapture(0)
        self.modeFunction = self.modestream
        self.tracker.deepsort.reset_tracker()
        
        

    # Botones
    def button_reproduce(self):
        self.MODE_VIDEO_REPRODUCE = True

    def button_pause(self):
        self.MODE_VIDEO_REPRODUCE = False


    #functions check boxes 
    def toggleDetector(self):
        self.typeDetector = not self.typeDetector
    def toggleTracker(self):
        self.typeTracker = not self.typeTracker 
    
            


if __name__=="__main__":
    root = tk.Tk()
    app = App(root)

