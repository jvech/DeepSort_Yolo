import time, random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


class DeepSort:
    def __init__(self,max_cosine_distance = 0.5,
                 nn_budget = None,nms_max_overlap = 1.0,
                 model_filename = 'model_data/mars-small128.pb',num_classes=80,
                 size = 416, classes = './data/labels/coco.names',
                 Tiny = False ,
                 weights = './weights/yolov3.tf',
                 detector = False):
        #constantes
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget 
        self.nms_max_overlap = nms_max_overlap 
        self.model_filename = model_filename
        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance,self.nn_budget)
        self.tracker = Tracker(self.metric)
        self.num_classes = num_classes
        self.size = size 
        
        #parametros detector
        self.classes = classes
        self.Tiny = Tiny 
        self.detector  = detector
        self.weights = weights 
        
        self.physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(self.physical_devices) > 0:
            tf.config.experimental.set_memory_growth(self.physical_devices[0], True)
            
        if self.Tiny:
            self.yolo = YoloV3Tiny(classes=self.num_classes)
        else:
            self.yolo = YoloV3(classes=self.num_classes)
        
        self.yolo.load_weights(self.weights).expect_partial()

        self.class_names = [c.strip() for c in open(self.classes).readlines()]
        
           
        
    def __call__(self,img):
        
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, self.size)
        
        boxes, scores, classes, nums = self.yolo.predict(img_in)
        
        # si solo queremos que funcione el detector 
        if self.detector:
            img = draw_outputs(img, (boxes, scores, classes, nums), self.class_names)
            return img, boxes
        
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(self.class_names[int(classes[i])])
            
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = self.encoder(img, converted_boxes)    
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)if class_name == "person"]
        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]        

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(img, class_name+ "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            

        return img, [ list(np.append(int(track.track_id), track.to_tlwh()))  for track in self.tracker.tracks if track.is_confirmed() or not track.time_since_update > 1]
 
#%%   

if __name__=="__main__":
    tracker = DeepSort()

    vid = cv2.VideoCapture("./videos_test/TUD-Stadtmitte.avi")

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('video.avi', codec, fps, (width, height))
    count = 0
    while True: 
        _, img = vid.read()

        if img is None:
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break
        img, tracks = tracker(img)
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

    vid.release()

    cv2.destroyAllWindows()









