"""
Module where all the GUI is created and handled
"""

import cv2
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from classDeepSort import DeepSort
from yolov3_tf2.utils import draw_YOLO, draw_DS
import os
from datetime import datetime


from  measure import Measure
from postProcessing import processing_csv 


class App:
    """ Class to handle the GUI
    """

    def __init__(self, master):
        # Constantes
        self.IMG_WIDTH = 640
        self.IMG_HEIGHT = 480
        self.FPS = 25
        
        self.system = System()

        #Ventana principal
        self.window = master
        self.window.title("Tradet-Net")
        master.attributes('-zoomed', True)

        #icono
        photo = tk.PhotoImage(file = os.path.join("data", "logo.png"))
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
        self.FrameRightPlus = tk.Frame(master)
        self.FrameLeft.pack(side=tk.LEFT, padx=5)
        self.FrameRight.pack(side=tk.LEFT)
        self.FrameRightPlus.pack(side=tk.LEFT)

        # Botones
        self.ButtonReproduce = tk.Button(self.FrameLeft, 
                                         command= self.button_reproduce,
                                         text="PLAY") # 
        self.ButtonRecord = tk.Button(self.FrameLeft, text="START RECORDING",
                                      command= self.button_record ) # 壘 
        self.ResetVideo = tk.Button(self.FrameLeft, text="RESET VIDEO", command= self.resetvideo) # 壘 
        self.ButtonPostProcess = tk.Button(self.FrameLeft,command=self.postprocess,text='GENERATE HEAT MAPS') 
        self.ButtonReproduce.grid(row=1, column=0,columnspan=2, padx=5, pady=5)
        self.ButtonRecord.grid(row=2, column=0,columnspan=2, padx=5, pady=5) 
        self.ResetVideo.grid(row=3, column=0,columnspan=2, padx=5, pady=5)
        self.ButtonPostProcess.grid(row=4, column=0,columnspan=2, padx=5, pady=5)

        self.MODE_VIDEO_REPRODUCE = False   

        # Stream Options
        self.MODE_STREAM_RECORD = False

        # Canvas fro principal Image
        self.CanvasMainImage = tk.Canvas(
                                    self.FrameRight, 
                                    width=self.IMG_WIDTH, 
                                    height=self.IMG_HEIGHT)
        self.CanvasMainImage.grid(
                row=0, column=0,
                padx=10, pady=10,
                sticky=tk.W + tk.S,
                )
        figure = plt.Figure(figsize=(5,4), dpi=100)
        self.ax = figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(figure, master)
        self.canvas.get_tk_widget().pack(side=tk.LEFT)#, fill=tk.BOTH)
        self.ax.set_title('Position')
        self.ax.set_xlabel('x meters')
        self.ax.set_ylabel('y meters')
        self.ax.grid()


        self.update()
        self.window.mainloop()

        try:
            self.window.destroy()
        except:
            pass

    def check(self):
        """ Check which objects is required to track, these work only in the extenden version
        """
        if self.ischecked[-2].get()==1 or self.ischecked[-1].get()==1:
            for check in self.checkboxes[:-1]:
                if self.ischecked[-2].get()==1:#all is  activated 
                    check.select()
                if self.ischecked[-1].get()==1:
                    check.deselect()
        self.system.objects = [name for i,name in enumerate(self.names) if self.ischecked[i].get()==1 ]
        if self.system.typeSource == "IMAGE":
            self.system.frameindex = 0

    def update(self): 
        """ Updates the visualization, the video and the plot of distances
        """
        if self.MODE_VIDEO_REPRODUCE or self.system.frameindex == 0:
            self.system()
        
        if self.system.typeSource == 'IMAGE':
            image = self.system.drawDetector()
        else:
            image = self.system.drawTracker()
            if len(self.system.pointBirds) > 0:
                self.ax.scatter(self.system.pointBirds[:,0],self.system.pointBirds[:,1])
                self.ax.plot(0,0,'r.')
                for i,txt in enumerate(self.system.id_ds):
                    self.ax.annotate(str(txt),(self.system.pointBirds[i,0],self.system.pointBirds[i,1]))
                self.ax.grid()
                self.ax.set_xlabel('x meters')
                self.ax.set_ylabel('y meters')
                self.canvas.draw()
                self.ax.clear()
                self.ax.set_title('Position')

            
        self.photo = cv2.resize(image, 
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
                    initialdir=os.path.join("."),
                    title="Select File",
                    filetypes = (
                        ("jpg files", "*.jpg"), 
                        ("png files", "*.png"),))
                        
        if filepath[-3:] in ["jpg", "png"]:
            self.system.reset(source=filepath, typeSource = 'IMAGE')
            
    def filemenu_openv(self):
        """Open Video"""
        self.MODE_VIDEO_REPRODUCE = False 
        self.filepath = filedialog.askopenfilename(
                        initialdir=os.path.join("."),
                        title="Select File", 
                        filetypes = (
                            ("avi files", "*.avi"),
                            ("mp4 files", "*.mp4"),))
                        
        if self.filepath[-3:] in ["avi", "mp4"]:
            self.system.reset(source=cv2.VideoCapture(self.filepath),typeSource = 'VIDEO')


    def filemenu_opens(self):
        """Open Stream"""
        self.MODE_VIDEO_REPRODUCE = True
        caption = cv2.VideoCapture(0)
        self.system.reset(source=caption, typeSource = 'STREAM')

    def postprocess(self):
        """ For generating the postprocess using the csv
        """
        self.ButtonPostProcess.config(text='GENERATING...')
        filepath = filedialog.askopenfilename(
                        initialdir=os.path.join("."),
                        title="Select File", 
                        filetypes = (
                            ("csv files", "*.csv"),))
        try:
            processing_csv(path_data=filepath)
            tk.messagebox.showinfo("INFO", "The heat maps have been created")
        except:
            tk.messagebox.showinfo("INFO", "You did not select a file")
        
        self.ButtonPostProcess.config(text='GENERATE HEAT MAPS')

    ## Buttons
    def button_reproduce(self):
        """ Button To reproduce the video
        """
        if self.system.source != None:
            self.ButtonReproduce.config(text="PAUSE" if not self.MODE_VIDEO_REPRODUCE else "PLAY")
            self.MODE_VIDEO_REPRODUCE = not self.MODE_VIDEO_REPRODUCE

    def button_record(self):
        """ Button to record the output video and csv
        """
        if self.system.source != None:
            self.system.realeaseFile()
            if self.system.SAVE == False:
                self.system.initSave()
            
            if self.system.SAVE == False and self.system.typeSource == 'IMAGE':
                self.system.save()
                self.system.realeaseFile()
                self.system.SAVE = not  self.system.SAVE
            
            self.system.SAVE = not  self.system.SAVE
            self.ButtonRecord.config(text="STOP RECORDING" if self.system.SAVE else "START RECORDING")

    def resetvideo(self):
        """ Button to reset the video
        """
        if self.system.typeSource == "VIDEO":
            self.ButtonReproduce.config(text="PLAY")
            self.MODE_VIDEO_REPRODUCE = False
            self.system.reset(source=cv2.VideoCapture(self.filepath),typeSource = 'VIDEO')



class System:
    """ CLass than handles the tracker system, save files, inputs and ouputs
    """
    def __init__(self):
        self.tracker = DeepSort()
        self.frame = cv2.imread(os.path.join("data","empty.jpeg"))
        self.frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)
        self.empty = False
        self.SAVE = False
        self.frameindex = 0
        self.typeSource = None
        self.source = None
        self.objects = "person"
        self.pointBirds= []
        
    def reset(self,source,typeSource):
        """ Takes the system to empty system
        """
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
        """ Load the data source
        """
        if self.typeSource == "IMAGE":
            if self.frameindex == 0:
                return True , cv2.imread(self.source)
            else: 
                return False, None  
        else:
            return self.source.read()
                    
    
    def __call__(self):
        """ call for get the detection and tracking objects, also save if it's required"""
        try:
            self.empty, imagen = self.loadSource()
            if self.empty == True:
                self.frame = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                self.boxes_ds, self.id_ds ,self.boxes,self.sco, self.classIDs, self.ids_nms, self.scales_nms, self.class_names  = self.tracker(self.frame,self.objects)
                self.frameindex += 1
                

                if self.frameindex == 1 and self.typeSource != 'IMAGE':
                    tk.messagebox.showinfo("INFO", "You must select four points that form a square in the real world, all sides must be 2 meters long. Also, this square must be on the plane where all the objects are going to walk.")
                    self.measureD = Measure(self.frame)
                if self.frameindex >1 and self.typeSource != 'IMAGE':
                   self.pointBirds,(Ox,Oy),(Xx,Xy),(Yx,Yy) = self.measureD(self.boxes_ds)
                   cv2.circle(self.frame, (Ox,Oy), 5, (255, 0, 0), 5)
                   cv2.arrowedLine(self.frame, (Ox,Oy), (Xx,Xy),(255,255,0), 2) 
                   cv2.putText(self.frame,'X',(Xx,Xy), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
                   cv2.arrowedLine(self.frame, (Ox,Oy), (Yx,Yy),(255,255,0), 2) 
                   cv2.putText(self.frame,'Y',(Yx,Yy), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)

                if self.SAVE == True:
                    self.save()
            else: 
                self.releaseFiles()
                
        except: 
            pass
    
    def drawDetector(self):
        """ Modify the image to show the detected boxes
        """
        if self.empty == True:
            return draw_YOLO(self.frame, (self.boxes, self.sco, self.classIDs, self.ids_nms, # para pintar el detector 
                              self.scales_nms), self.class_names)
        else:
            return self.frame
        
    def drawTracker(self):
        """Modify the image to show the tracked objects
        """
        if self.empty == True:
            return draw_DS(self.frame, self.boxes_ds, self.id_ds)
        else:
            return self.frame
            
    def drawtwo(self):
        """ Combines drawTracker and drawDetector
        """
        if self.empty == True : 
            return draw_YOLO(draw_DS(self.frame, self.boxes_ds, self.id_ds), (self.boxes, self.sco, self.classIDs, self.ids_nms, 
                              self.scales_nms), self.class_names)
        else: 
            return self.frame 
        
    def save(self):
        """ Save video or image
        """
        self.save_annotations()
        if self.typeSource != 'IMAGE':
            self.out_video.write(cv2.cvtColor(self.drawTracker(), cv2.COLOR_BGR2RGB))
        else:
            cv2.imwrite(os.path.join("output",str(datetime.now())[:-7]+".png"),cv2.cvtColor(self.drawDetector(), cv2.COLOR_BGR2RGB)) 
            
        
    def save_annotations(self):
        """ Save the csv for detection and tracking
        """
        if self.typeSource == 'IMAGE':
            self.boxes_ds = self.boxes 
            self.id_ds = np.zeros((1,len(self.boxes)))        
            self.pointBirds = np.zeros((2,len(self.boxes)))
        
        for bbox, id_, scor, classId,Bxy in zip(self.boxes_ds, self.id_ds, self.sco, self.classIDs,self.pointBirds):
            x, y = bbox[0], bbox[1]
            w, h = bbox[2] - x, bbox[3] - y
            score = int(100*scor)
            self.annotations_file.write(
                    "%d,%d,%d,%d,%d,%d,%d,%d,%f,%f\n"\
                    %(self.frameindex, int(id_), x, y, w, h, score, classId,Bxy[0],Bxy[1])
                    )
    
    def realeaseFile(self):
        """ release the ouputs files
        """
        try:
            self.annotations_file.close()
            self.out_video.release()
        except:
            pass
        
    def initSave(self):
        """" Initializer the ouput files
        """
        self.realeaseFile()
        time_mark = str(datetime.now())[:-7].replace(' ','_').replace('-','_')
        path_parent = os.path.join('output',time_mark)
        os.mkdir(path_parent) 
        path = os.path.join(path_parent,time_mark+".csv")
        self.annotations_file = open(path, "w")
        self.annotations_file.write("frame,id,x,y,w,h,score,class,Bx,By\n")
                   
        if self.typeSource != 'IMAGE':
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_path = path[:-3]+"avi"
            fps = self.source.get(cv2.CAP_PROP_FPS)
            self.out_video = cv2.VideoWriter(video_path, fourcc, fps, 
                                                (int(self.source.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.source.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                        

if __name__=="__main__":
    root = tk.Tk()
    app = App(root)

