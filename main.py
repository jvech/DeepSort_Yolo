import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from classDeepSort import DeepSort
from yolov3_tf2.utils import draw_YOLO, draw_DS
import os 
from datetime import datetime


"""
Cosas por hacer:

-> actualizar la funciones draw_YOLO para que muestre clase y score 

-> agregar boton de stop para cerrar los archivos a guardar 

"""


class App:
    def __init__(self, master):
        # Constantes
        self.IMG_WIDTH = 640
        self.IMG_HEIGHT = 480
        self.FPS = 25

        #Ventana principal
        self.window = master
        self.window.title = "MAIN WINDOW"

        #icono
        photo = tk.PhotoImage(file = "./data/logo.png")
        self.window.iconphoto(False, photo)

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
        self.ButtonRecord = tk.Button(self.FrameLeft, text="RECORD",
                                      command=self.button_record) # 壘

        self.ButtonReproduce.grid(row=1, column=0, padx=5, pady=5)
        self.ButtonPause.grid(row=1, column=1, padx=5, pady=5)
        self.ButtonRecord.grid(row=2, column=0, padx=5, pady=5)
        
        #self.ButtonStop.grid(row=0, column=2, padx=5, pady=5)
        
        #Check Buttons
        self.checkboxDec = tk.Checkbutton(self.FrameLeft, text="Detector",
                                          command=self.toggleDetector)
        self.checkboxTra = tk.Checkbutton(self.FrameLeft, text="Tracker",
                                          command= self.toggleTracker)

        self.checkboxDec.grid(row=0, column=0, padx=5, pady=5)
        self.checkboxTra.grid(row=0, column=1, padx=5, pady=5)

        #self.checkboxDec.select()
        #self.checkboxTra.select()        

        # Image Options
        # Video Options
        self.MODE_VIDEO_REPRODUCE = True  

        # Stream Options
        self.MODE_STREAM_RECORD = False

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
        self.frameindex = 0 
        self.filepath = os.path.join('data','logo.png')
        self.annotations_file = None
        self.out_video = None

        self.tracker = DeepSort() 
        self.typeTracker = False # True si se quiere usar como tracker
        self.typeDetector = False #True si se quiere usar como detector           

        self.mode_function = self.mode_image #por default 

        # Main
        self.update()
        self.window.mainloop()
        
        #all the things that you need to close put here 
         
        # Solución problema de cierre de ventana 
        try:
            self.window.destroy()
        except:
            pass

    def update(self):
        """Función para actualizar en screen
        poner todo lo que se requiera actualizar"""   
        
        
        # mode_function: imagen, video o streaming
        # actializa frame cargando desde la fuente  
        if self.MODE_VIDEO_REPRODUCE:
            self.mode_function()
            self.trackdetec()
            self.frameindex += 1
            if self.mode_function == self.mode_image:
            	self.MODE_VIDEO_REPRODUCE = True 
         
        self.photo = cv2.resize(self.frame, 
                                dsize=(self.IMG_WIDTH, self.IMG_HEIGHT), 
                                interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.photo))
        
        self.CanvasMainImage.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(int(1000/self.FPS), self.update)

    def trackdetec(self):
        """Actualiza frame con detecciones o tracks """
        frame = cv2.cvtColor(self.frame.copy(), cv2.COLOR_BGR2RGB)
        if self.typeDetector or self.typeTracker: 
            boxes_ds, id_ds ,boxes_nms,\
            sco_nms, classIDs_nms, ids_nms,\
            scales_nms, class_names  = self.tracker(self.frame)

        if self.typeTracker:    
            self.frame = draw_DS(self.frame, boxes_ds, id_ds)    # para pintar el tracker
        
        if self.typeDetector:
            self.frame = draw_YOLO(self.frame, (boxes_nms, sco_nms, classIDs_nms, ids_nms, # para pintar el detector 
                              scales_nms), class_names)
        if self.MODE_STREAM_RECORD and self.mode_function != self.mode_image:
            self.save_annotations(boxes_ds, id_ds, sco_nms, classIDs_nms)
            self.out_video.write(frame)
            
            
       	if  self.MODE_STREAM_RECORD and self.mode_function == self.mode_image and self.frameindex ==0:
       		self.save_annotations(boxes_ds, id_ds, sco_nms, classIDs_nms)
       		path = os.path.join("output",str(datetime.now())[:-7]+".jpg")
       		cv2.imwrite(path,cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
			
       	
            
    def mode_image(self):
        """Actualiza frame con imagen cargada"""
        self.FPS = 25
        try:
            self.caption.release()
        except AttributeError:
            pass
        self.frame = cv2.imread(self.filepath)
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

    def mode_video(self):
        self.FPS = self.caption.get(cv2.CAP_PROP_FPS)
        ret, frame = self.video_handler()
        if ret == True:
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            self.MODE_VIDEO_REPRODUCE = False 

    def mode_stream(self):
        self.FPS = 25
        ret, self.frame = self.caption.read()
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

    def video_handler(self):
        if self.MODE_VIDEO_REPRODUCE or self.photo == None:
            return self.caption.read()
        else:
            return True, cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)


    # Funciones widgets 

    ## Menus
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
            self.mode_function =  self.mode_image
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
            self.mode_function =  self.mode_video
            self.tracker.deepsort.reset_tracker()
            self.frameindex = 0

    def filemenu_opens(self):
        """Open Stream"""
        self.MODE_VIDEO_REPRODUCE = True
        self.caption = cv2.VideoCapture(0)
        self.mode_function = self.mode_stream
        self.tracker.deepsort.reset_tracker()
        self.frameindex = 0

    ## Buttons
    def button_reproduce(self):
        self.MODE_VIDEO_REPRODUCE = True

    def button_pause(self):
        self.MODE_VIDEO_REPRODUCE = False

    def button_record(self):
        if self.typeTracker:
            self.MODE_STREAM_RECORD = not self.MODE_STREAM_RECORD
            self.frameindex = 0
            if self.MODE_STREAM_RECORD:
                self.ButtonRecord.config(text="RECORDING")
                path = os.path.join("output",str(datetime.now())[:-7]+".csv")
                self.annotations_file = open(path, "w")
                self.annotations_file.write(
                        "frame,id,x,y,w,h,score,class\n"
                        )
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                if self.mode_function != self.mode_image:
                    video_path = path[:-3]+"avi"
                    self.out_video = cv2.VideoWriter(video_path, fourcc, self.FPS, 
                                                (self.IMG_WIDTH, self.IMG_HEIGHT))
            else:
                self.annotations_file.close()
                if self.mode_function != self.mode_image:               
                	self.out_video.release()
                	self.out_video = None
                self.ButtonRecord.config(text="RECORD")
     


    ## Checkboxes
    def toggleDetector(self):
        self.typeDetector = not self.typeDetector
    def toggleTracker(self):
        self.typeTracker = not self.typeTracker 

    # Save Part
    def save_annotations(self, boxes, ids, sco, classIDs):
        for bbox, id_, scor, classId in zip(boxes, ids, sco, classIDs):
            x, y = bbox[0], bbox[1]
            w, h = bbox[2] - x, bbox[3] - y
            score = int(100*scor)
            self.annotations_file.write(
                    "%d,%d,%d,%d,%d,%d,%d,%s\n"\
                    %(self.frameindex, int(id_), x, y, w, h, score, classId)
                    )

    def save_video(self):
        pass


if __name__=="__main__":
    root = tk.Tk()
    app = App(root)

