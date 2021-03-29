"""
17/3/2021
Module to measure the distance between objects using
bird's eye view from opencv
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2


def get_mouse_points(event, x, y, flags, param):
    """get the four  points from image using mouse"""
    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        if len(mouse_pts) == 0:
            cv2.putText(image,'origin',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
        elif len(mouse_pts) == 1:
            cv2.putText(image,'X',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
        elif len(mouse_pts) == 3:  
            cv2.putText(image,'Y',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)

        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
            cv2.line(image, (x, y),
                    (mouse_pts[len(mouse_pts)-1][0],
                    mouse_pts[len(mouse_pts)-1][1]),
                    (70, 70, 70), 2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y),
                (mouse_pts[0][0], mouse_pts[0][1]),
                (70, 70, 70), 2)
        mouse_pts.append((x, y))

def boxes_to_points(boxes):
    points = []
    for box in boxes:
        (x,y) = (box[2]-(box[2]-box[0])/2,box[3])
        points.append([x,y])
    return  points

class Measure:
    def __init__(self, frame,dis_x=2,dis_y=2,number_points=5):
        """
        Init the referencial points to stimate homography
        matrix
        """
        global mouse_pts  # to use with mouse
        global image #to use with mouse 
        self.wh = np.flip(frame.shape[:2])
        self.dis_x =dis_x # mts 
        self.dis_y =dis_y # mts
        image = np.copy(frame)
        mouse_pts = []
        cv2.namedWindow('imagen', cv2.WINDOW_NORMAL)
        #cv2.setWindowProperty('imagen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback('imagen',get_mouse_points)
        while cv2.getWindowProperty('imagen', cv2.WND_PROP_VISIBLE) > 0:
            if len(mouse_pts) < number_points:
                cv2.imshow('imagen',image)
                cv2.waitKey(1)
            else:
                cv2.destroyWindow('imagen')
                break
        image = frame
        if len(mouse_pts) < number_points:
            mouse_pts = [[0,0],[frame.shape[1],0],[frame.shape[1],frame.shape[0]],[0,frame.shape[0]]]
        self.generate_birds_view(mouse_pts[:4])
        self.unit_length()

    def __call__(self,boxes):
        """from bounding box obtain the point in the middle of the of bottom
        line and map to the birds view
        """
        points = boxes_to_points(boxes)
        birdsPont = self.plot_point_to_birds_view(points)
        return birdsPont,mouse_pts[0],mouse_pts[1],mouse_pts[3]

    def generate_birds_view(self,points):
        """with the points selected we create new imagen(bird's eye view
        where we measure the distance"""
        (tl, tr, br, bl) = points
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        self.maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        self.maxHeight = max(int(heightA), int(heightB))
        dts = np.array([[0, 0],
                [self.maxWidth , 0],
                [self.maxWidth,self.maxHeight],
                [0,self.maxHeight]], dtype = "float32")
        points = np.float32(np.array(points))
        self.Matrix = cv2.getPerspectiveTransform(points,dts)
        self.image=self.templateIMG = cv2.warpPerspective(image,self.Matrix,(self.maxWidth,self.maxHeight))

    def transform_point_to_birds_view(self,points):
        """transform the point to birds plane"""
        birds_point = cv2.perspectiveTransform(points,self.Matrix)
        return birds_point[0][0]

    def plot_point_to_birds_view(self,points):
        birdsPont = []
        for  point in points:
            (xb,yb) = self.transform_point_to_birds_view(np.float32(np.array([[point]])))
            birdsPont.append([xb/self.dw*self.dis_x,yb/self.dh*self.dis_y])
        return np.array(birdsPont)

    def plot_image(self,Birdspoints):
        self.image = np.copy(self.templateIMG)
        for point in Birdspoints:
            cv2.circle(self.image, (point[0], point[1]), 5, (0, 0, 255), 10)
        cv2.imshow('image',self.image)
        cv2.waitKey(1)

    def unit_length(self):
        points = np.float32(np.array([[mouse_pts[0],mouse_pts[1],mouse_pts[3]]]))
        warped_pt= cv2.perspectiveTransform(points,self.Matrix)[0]
        self.dw = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        self.dh= np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)

    def distance_real_world(self,point1,point2):
        (x1,y1) = point1
        (x2,y2) = point2
        return np.sqrt((((x1-x2)/self.dw)*self.dis_x)**2 + ((((y1-y2)/self.dh)*self.dis_y)**2))


if __name__=="__main__":
   image = cv2.imread('imagen.jpeg')
   img = image
   #measureD = Measure(image,dis_x=6.1,dis_y=6.3,number_points=7)
   measureD = Measure(image,dis_x=7.2,dis_y=8.1,number_points=7)
   measureD.unit_length()
   (x1,y1) = measureD.transform_point_to_birds_view(np.float32(np.array([[mouse_pts[4]]])))
   (x2,y2) = measureD.transform_point_to_birds_view(np.float32(np.array([[mouse_pts[5]]]))) 
   distance =measureD.distance_real_world((x1,y1),(x2,y2))
   cv2.line(img, mouse_pts[4] ,mouse_pts[5],  (0, 255, 0) , 9) 
   cv2.putText(img,str(distance),mouse_pts[5], cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
   plt.subplot(121),plt.imshow(img),plt.title('Input')
   plt.subplot(122),plt.imshow(measureD.image),plt.title('Output')
   plt.show()


