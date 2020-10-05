'''
DeepSort Yolo algorithm in sklearn/tensorflow keras.
'''


'''
Que falta:

    3) determinar el tamaño del hypercolumn
'''
from absl import flags
from absl.flags import FLAGS
import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_YOLO, draw_DS, draw_output
import imutils
from deepsort import deepsort_rbc
import warnings
warnings.filterwarnings('ignore')


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


flags.DEFINE_string('video',"./videos_test/TUD-Stadtmitte.avi",
                     'path to video file')
flags.DEFINE_string('classes', './data/coco.names',
                     'path to file with db names')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                     'path to file with network name')
flags.DEFINE_integer('size', 416, 'size of network input')
flags.DEFINE_integer('num_classes', 80, 'Number of classes to recognize')
flags.DEFINE_boolean('save', False, 'if true, saves yolo-features')
flags.DEFINE_string('output', './data/output.avi',
                     'path to output video file')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_boolean('show', True, 'if true, show the video') 



# Main code
def main(_argv):
    # try access to GPU
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # Create Yolo model (Tiny or complete)
    if FLAGS.tiny:
        # light version (faster processing)
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        # Deeper more robust model
        yolo = YoloV3(classes=FLAGS.num_classes)

    # Load weights from a pretrained net
    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    # Get classes names
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    #Initialize deep sort.
    deepsort = deepsort_rbc()

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        
        list_file = open('detection.txt', 'w') #######
        
    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vid.get(prop))
        logging.info("{} total frames in video".format(total))
        # print("[INFO] {} total frames in video".format(total))

    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        logging.info("could not determine # of frames in video")
        logging.info("No approx. completion time can be provided")
        total = -1

    # number of frames counter
    cont = 0

    # Write Yolo features from each frame to `images.tfrecords`.
    #record_file = 'Test.tfrecords'
    #with tf.io.TFRecordWriter(record_file) as writer:
    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            break

        # print(cont)
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()

        # use model to predict object bboxes
        t1 = time.time()
        ind, boxes, scores, classes, nums, feat = yolo(img_in)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]
        fps = 1.0 / (t2- t1)	
        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
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
                    if class_names[classID] == "person" or class_names[classID] == "person":
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

        # save output image
        # img2 = draw_output(img, (bboxes_p,confidences,classIDs,ids,
        #                       scales), class_names)
        # cv2.imshow('output', img2)
        # key = cv2.waitKey(0) & 0xFF
        # if key == ord('q'):
        #     break
        if FLAGS.save:
            # Save features to TFRecord
            if feat_all:
                t_feat_all = tf.convert_to_tensor(feat_all)
                # Process the frames into `tf.Example` messages.
                tf_example = frame_example(t_feat_all, cont)
                # Write to a `.tfrecords` file.
                writer.write(tf_example.SerializeToString())
               

        # ensure at least one detection exists
        if bboxes_p:

            # if cont == 46:
            #     print(cont)
            # feed deepsort with detections and features
            tracker, detections_class = deepsort.run_deep_sort(img, np.asarray(bboxes_p),
                                                              confidences,
                                                              classIDs, scales,
                                                              ids, feat_track)
            classIDs_nms = detections_class[1]
            scales_nms = detections_class[2]
            ids_nms = detections_class[3]
            # prelocation employed detections
            boxes_nms = []
            sco_nms = []
            for det in detections_class[0]:
                # Append NMS detection boxes
                boxes_nms.append(det.to_tlbr())
                sco_nms.append(det.confidence)

            # prelocation of tracked boxes
            boxes_ds = []
            id_ds = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                #Append boxes and id.
                boxes_ds.append(track.to_tlbr())   #Get the corrected/predicted bounding box
                id_ds.append(str(track.track_id))  #Get the ID for the particular track.
                #poner aqui la generación del txt para medir el rendimiento s

        # save output image
        img = draw_YOLO(img, (boxes_nms, sco_nms, classIDs_nms, ids_nms,
                              scales_nms), class_names)
        if boxes_ds:
            img = draw_DS(img, boxes_ds, id_ds)
        img = cv2.putText(img, "Time: {:.2f}ms, frame:{:d}".format(sum(times)/len(times)*1000, cont), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                          
        print("FPS: %.2f" % fps)
        
        
        boxes_tracking = np.array([track.to_tlwh() for track in tracker.tracks])
        identificadores = np.array([track.track_id for track in tracker.tracks])
        #clas_ = np.array([track.get_class() for track in tracker.tracks])
        #boxes_tracking = boxes_tracking[clas_=='person']
        #identificadores = identificadores[clas_=='person']
        
        
        if FLAGS.output:
            out.write(img)
            
            #list_file.write(str(frame_index)+' ')
            if len(boxes_tracking) != 0:
                for i in range(0,len(boxes_tracking)):
                    list_file.write(str(cont)+' '+ str(identificadores[i]) + ' '+str(boxes_tracking[i][0]) + ' '+str(boxes_tracking[i][1]) + ' '+str(boxes_tracking[i][2]) + ' '+str(boxes_tracking[i][3]) + ' ')
                    list_file.write('\n')
                    
        if FLAGS.show:
            cv2.imshow('output', img)
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            cont += 1
    cv2.destroyAllWindows()

# Initialize code
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
