import cv2
from .object_detection import Model
import numpy as np

import os
import io
import PIL.Image as Image
import torch
import winsound

class VideoCamera(object):
    def __init__(self):
   
        self.video = cv2.VideoCapture(0)
        self.yolo = Model.YoloModel()
        self.model = self.yolo.load_model()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
 
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()

        x_c,y_c = self.yolo.detect(image,self.model)

        for i in range(65):
            cv2.line(image, (i*10, 0), (i*10, 500), (0, 255, 0), 1, 1)
            cv2.line(image, (0,i*10), (1000,i*10), (0, 255, 0), 1, 1)


        cv2.circle(image,(int(image.shape[1]/2),int(image.shape[0]/2)),10,(255, 100, 0),2)

        if((x_c-(int(image.shape[1]/2+20))>0)):
            # print("Here\n")
            cv2.arrowedLine(image,(int(image.shape[1]/2+20),int(image.shape[0]/2)) , (int(image.shape[1]/2+60),int(image.shape[0]/2)),
                                     (255,0,0), 10,tipLength=0.5)

        if((x_c-(int(image.shape[1]/2-20))<0)):
            # print("Here\n")
            cv2.arrowedLine(image,(int(image.shape[1]/2-20),(int(image.shape[0]/2))),((int(image.shape[1]/2-60)) ,int(image.shape[0]/2)),
                                     (255,0,0), 10,tipLength=0.5)
        

        if((y_c-(int(image.shape[0]/2+20))>0)):
            # print("Here\n")
            cv2.arrowedLine(image,(int(image.shape[1]/2),int(image.shape[0]/2+20)) , (int(image.shape[1]/2),int(image.shape[0]/2+60)),
                                     (255,0,0), 10,tipLength=0.5)

        if((y_c-(int(image.shape[0]/2-20))<0)):
            # print("Here\n")
            cv2.arrowedLine(image,(int(image.shape[1]/2),(int(image.shape[0]/2-20))),((int(image.shape[1]/2)) ,int(image.shape[0]/2-60)),
                                     (255,0,0), 10,tipLength=0.5)


        if(int(image.shape[1]/2)-20< x_c <int(image.shape[1]/2)+20 and int(image.shape[0]/2)-20< y_c <int(image.shape[0]/2)+20):
            # winsound.Beep(440, 500)
            cv2.circle(image,(int(image.shape[1]/2),int(image.shape[0]/2)),20,(0, 0, 255),-1)
            cv2.circle(image,(int(image.shape[1]/2),int(image.shape[0]/2)),30,(0, 0, 255),1)
            winsound.Beep(600, 200)
            

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()

        # frame = np.array(frame)
        # image = Image.open(io.BytesIO(bytes))

        # yolo = Model.YoloModel()

        # model = yolo.load_model()

        # yolo.detect(frame,model)

        

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')