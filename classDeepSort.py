import time 
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_YOLO, draw_DS
import imutils
from deepsort import deepsort_rbc
import warnings
warnings.filterwarnings('ignore')
from os import path


# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# tf.example definition with relevant features.
def frame_example(raw, frame):

  feature = {
      'frame': _int64_feature(frame),
      'raw_features': _bytes_feature(tf.io.serialize_tensor(raw)),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))


class DeepSort: 
    def __init__(self,num_classes=80,
                 size = 416, classes = path.join("data","coco.names"),
                 detectorTiny = False ,
                 weights = path.join('.','data','yolov3_model','yolov3.tf')):
                 
        #constantes
        self.num_classes = num_classes
        self.size = size 
 
        #parametros detector
        self.classes = classes
        self.detectorTiny = detectorTiny 
        self.weights = weights         
        self.deepsort = deepsort_rbc()
        
        self.physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(self.physical_devices) > 0:
            tf.config.experimental.set_memory_growth(self.physical_devices[0], True)
            
        if self.detectorTiny:
            self.yolo = YoloV3Tiny(classes=self.num_classes)
        else:
            self.yolo = YoloV3(classes=self.num_classes)
        
        self.yolo.load_weights(self.weights).expect_partial()

        self.class_names = [c.strip() for c in open(self.classes).readlines()]        

    def __call__(self,img,objects):
        """ parámetros: imagen 
            Salida: boxes_ds, id_ds ,boxes, sco, classIDs, ids, scales, class_names
             
            boxes_ds y id_ds salidas del tracker 
            boxes, sco, classIDs, ids, scales, class_names son salidas del detector
            
            
            sco: score detecciones 
            classIDs: entero que define a que clase pertenece la detección 
            ids: entero que define el orden en que se realizaron las detecciones en cada frame 
            scales: define cual salida de la yolo realizo la detección, recordar que se tienen tres salidas cada una define una escala 
            class_names: es un vector de string que tiene los nombres de las clases 
            """
    
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, self.size)
        ind, boxes, scores, classes, nums, feat = self.yolo(img_in)
        

        bboxes_p = []
        confidences = []
        classIDs = []
        feat_track = []
        scales = []
        ids = []

        # feature pre-location
        feat_all = []
        conti = 0

        for j, indi in enumerate(ind):
            if indi is not None:
                for i in indi:
                    classID = int(classes[0][int(i[4])])
                    if self.class_names[classID] in objects:
                        sco = np.array(scores[0][int(i[4])])
                        box_p = np.array(boxes[0][int(i[4])])
                        # logging.info('\t{}, {}, {}'.format(class_names[classID],
                        #                                    sco,box_p))

                        # Feature extraction
                        x, y = np.array(i[1:3])
                        feat_1 = feat[j][:, x, y, :][0]
                        feat_track.append(feat_1)
                        feat_all.append(np.concatenate([feat_1,
                                              tf.expand_dims(classes[0][int(i[4])], 0), # object class
                                              tf.expand_dims(i[4], 0)], axis=0)) # id object in frame

                        # objects allocation
                        ids.append(conti)
                        conti += 1
                        scales.append(j)
                        confidences.append(float(sco))
                        bboxes_p.append(box_p)
                        classIDs.append(classID)


        classIDs_nms = []
        scales_nms = []
        ids_nms = []
        boxes_nms = []
        sco_nms = []
        boxes_ds = []
        id_ds = []
        # ensure at least one detection exists
        if bboxes_p:

            # if cont == 46:
            #     print(cont)
            # feed deepsort with detections and features
            tracker, detections_class = self.deepsort.run_deep_sort(img, np.asarray(bboxes_p),
                                                              confidences,
                                                              classIDs, scales,
                                                              ids, feat_track)
            classIDs_nms = detections_class[1]
            scales_nms = detections_class[2]
            ids_nms = detections_class[3]
            # prelocation employed detections
            for det in detections_class[0]:
                # Append NMS detection boxes
                boxes_nms.append(det.to_tlbr())
                sco_nms.append(det.confidence)

            # prelocation of tracked boxes

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                #Append boxes and id.
                boxes_ds.append(track.to_tlbr())   #Get the corrected/predicted bounding box
                id_ds.append(str(track.track_id))  #Get the ID for the particular track.
                
        return  boxes_ds, id_ds ,boxes_nms, sco_nms, classIDs_nms, ids_nms, scales_nms, self.class_names   



if __name__=="__main__":

    tracker = DeepSort()    

    vid = cv2.VideoCapture("./ETH-Sunnyday.avi")

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
        boxes_ds, id_ds ,boxes_nms, sco_nms, classIDs_nms, ids_nms, scales_nms, class_names  = tracker(img)
        for boxes,id_ in zip(boxes_ds,id_ds):
        	print(boxes,id_ ,end='\n')    
        img = draw_DS(img, boxes_ds, id_ds)    # para pintar el traker 
        img = draw_YOLO(img, (boxes_nms, sco_nms, classIDs_nms, ids_nms, # para pintar el detector 
                              scales_nms), class_names)
                              
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

