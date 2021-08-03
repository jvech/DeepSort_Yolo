""" Script that handles the deepsort algorithm
"""


from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker 
from deep_sort.application_util.preprocessing import non_max_suppression
from deep_sort.deep_sort.detection import Detection
import numpy as np


class deepsort_rbc():
    """ Class to handle the deepsort algorithm
    """
    def __init__(self):

        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine",.5,30) #metric, matching_threshold, budget
        self.tracker= Tracker(self.metric)


    def reset_tracker(self):
        self.tracker= Tracker(self.metric)

    #Deep sort needs the format `top_left_x, top_left_y, width,height
    
    def format_boxes(self,out_boxes):
        for b in range(len(out_boxes)):
            out_boxes[b][2] = out_boxes[b][2] - out_boxes[b][0]
            out_boxes[b][3] = out_boxes[b][3] - out_boxes[b][1]
        return out_boxes                

    def run_deep_sort(self,frame, out_boxes, out_scores, classIDs, 
                      scales, ids, features):

        if out_boxes==[]:            
            self.tracker.predict()
            print('No detections')
            trackers = self.tracker.tracks
            return trackers
        
        wh = np.flip(frame.shape[0:2])
        
        out_boxes[:,0:2] = (out_boxes[:,0:2] * wh).astype(float)
        out_boxes[:,2:4] = (out_boxes[:,2:4] * wh).astype(float)   
        
        # Give format boxes        
        out_boxes = self.format_boxes(out_boxes)
        
        # Create a detection object to parse in deepsort
        dets = [Detection(bbox, score, feature) \
                    for bbox, score, feature in\
                zip(out_boxes, out_scores, features)]

        outboxes = np.array([d.tlwh for d in dets])
        outscores = np.array([d.confidence for d in dets])
        
        # Non max suppression including confidence
        # works best using pixel coordinates, 0.3<=overlap<=0.5
        indices = non_max_suppression(outboxes,0.4,outscores)
        
        dets = [dets[i] for i in indices]
        classIDs = [classIDs[i] for i in indices]
        scales = [scales[i] for i in indices]
        ids = [ids[i] for i in indices]
        
        detections_class =(dets,classIDs,scales,ids)
        
        # DeepSort cycle
        self.tracker.predict()
        self.tracker.update(dets)    

        return self.tracker,detections_class



