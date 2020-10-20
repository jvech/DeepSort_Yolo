import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from classDeepSort import DeepSort
from yolov3_tf2.utils import draw_YOLO, draw_DS
import os 
from datetime import datetime


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
        self.ButtonPause = tk.Button(self.FrameLeft, 
                                     command=self.button_pause,
                                     text="PAUSE") # 
        self.ButtonRecord = tk.Button(self.FrameLeft, text="START RECORDING",
                                      command=self.button_record) # 壘

        self.ButtonReproduce.grid(row=1, column=0, padx=5, pady=5)
        self.ButtonPause.grid(row=1, column=1, padx=5, pady=5)
        self.ButtonRecord.grid(row=2, column=0, padx=5, pady=5)
        
        
        #Check Buttons
        self.checkboxDec = tk.Checkbutton(self.FrameLeft, text="Detector",
                                          command=self.toggleDetector)
        self.checkboxTra = tk.Checkbutton(self.FrameLeft, text="Tracker",
                                          command= self.toggleTracker)

        self.checkboxDec.grid(row=0, column=0, padx=5, pady=5)
        self.checkboxTra.grid(row=0, column=1, padx=5, pady=5)    

        # Image Options
        # Video Options
        self.MODE_VIDEO_REPRODUCE = False   

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
        self.system = System()
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
        if self.MODE_VIDEO_REPRODUCE or self.system.frameindex == 0:
            self.system()
            
        self.photo = cv2.resize(self.system.drawTracker(), 
                                dsize=(self.IMG_WIDTH, self.IMG_HEIGHT), 
                                interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.photo))
        self.CanvasMainImage.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(int(1000/self.FPS), self.update)


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
        	self.system.reset(source=filepath, typeSource = 'IMAGE')
        	

    def filemenu_openv(self):
        """Open Video"""
        filepath = filedialog.askopenfilename(
                    initialdir="./",
                    title="Select File", 
                    filetypes = (
                        ("avi files", "*.avi"),
                        ("mp4 files", "*.mp4"),))
                        
        if filepath[-3:] in ["avi", "mp4"]:
        	self.system.reset(source=cv2.VideoCapture(filepath),typeSource = 'VIDEO')

    def filemenu_opens(self):
        """Open Stream"""
        self.MODE_VIDEO_REPRODUCE = True
        caption = cv2.VideoCapture(0)
        self.system.reset(source=caption, typeSource = 'STREAM')


    ## Buttons
    def button_reproduce(self):
        self.MODE_VIDEO_REPRODUCE = True

    def button_pause(self):
        self.MODE_VIDEO_REPRODUCE = False

    def button_record(self):
    	if self.system.SAVE == False:
    		self.system.initSave()
    	self.system.SAVE = not  self.system.SAVE
    	self.ButtonRecord.config(text="STOP RECORDING" if self.system.SAVE else "START RECORDING")
    	
    ## Checkboxes
    def toggleDetector(self):
        self.typeDetector = not self.typeDetector
    def toggleTracker(self):
        self.typeTracker = not self.typeTracker 



class System:
	def __init__(self):
		self.tracker = DeepSort()
		self.frame = cv2.imread('./data/logo.png')
		self.empty = False
		self.SAVE = False
		self.frameindex = 0 
		
	def reset(self,source,typeSource):
		try:
			self.source.realease()
		except: 
			pass
		self.source = source  
		self.typeSource = typeSource
		self.tracker.deepsort.reset_tracker()
		self.frameindex = 0 
		self.SAVE = False 
			
	def loadSource(self):
		if self.typeSource == "IMAGE":
			if self.frameindex == 0:
				return True , cv2.imread(self.source)
			else: 
				return False, None  
		else:
			return self.source.read()
					
	
	def __call__(self,):
		"""función para calcular detecciónes y salvar si es el caso, 
		si es guardar imagen solo se hace una vez si es la misma imagen"""
		try:
			self.empty, image = self.loadSource()
			if self.empty == True:
				self.frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				self.boxes_ds, self.id_ds ,self.boxes,self.sco, self.classIDs, self.ids_nms, self.scales_nms, self.class_names  = self.tracker(self.frame)
				self.frameindex += 1 
				if self.SAVE == True:
					self.save()
				
		except: 
			pass 
	
	def drawDetector(self):
		if self.empty == True:
			return draw_YOLO(self.frame, (self.boxes, self.sco, self.classIDs, self.ids_nms, # para pintar el detector 
                              self.scales_nms), self.class_names)
		else:
			return self.frame
		
	def drawTracker(self):
		if self.empty == True:
			return draw_DS(self.frame, self.boxes_ds, self.id_ds)
		else:
			return self.frame
			
	def drawtwo(self):
		if self.empty == True : 
			return draw_YOLO(draw_DS(self.frame, self.boxes_ds, self.id_ds), (self.boxes, self.sco, self.classIDs, self.ids_nms, 
                              self.scales_nms), self.class_names)
		else: 
			return self.frame 
		
	def save(self):
		self.save_annotations()
		if self.typeSource != 'IMAGE':
			self.out_video.write(cv2.cvtColor(self.drawTracker(), cv2.COLOR_BGR2RGB))
		else:
			cv2.imwrite(os.path.join("output",str(datetime.now())[:-7]+".jpg"),cv2.cvtColor(self.drawDetector(), cv2.COLOR_BGR2RGB)) 
		
	def save_annotations(self):
		for bbox, id_, scor, classId in zip(self.boxes_ds, self.id_ds, self.sco, self.classIDs):
			x, y = bbox[0], bbox[1]
			w, h = bbox[2] - x, bbox[3] - y
			score = int(100*scor)
			self.annotations_file.write(
                    "%d,%d,%d,%d,%d,%d,%d,%d\n"\
                    %(self.frameindex, int(id_), x, y, w, h, score, classId)
                    )
		
	def initSave(self):
		try:
			self.annotations_file.close()
			self.out_video.release()
		except:
			pass
		path = os.path.join("output",str(datetime.now())[:-7]+".csv")
		self.annotations_file = open(path, "w")
		self.annotations_file.write("frame,id,x,y,w,h,score,class\n")
                        
		if self.typeSource != 'IMAGE':
			fourcc = cv2.VideoWriter_fourcc(*'MJPG')
			video_path = path[:-3]+"avi"
			fps = self.source.get(cv2.CAP_PROP_FPS)
			self.out_video = cv2.VideoWriter(video_path, fourcc, fps, 
                                                (int(self.source.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.source.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                        

if __name__=="__main__":
    root = tk.Tk()
    app = App(root)

