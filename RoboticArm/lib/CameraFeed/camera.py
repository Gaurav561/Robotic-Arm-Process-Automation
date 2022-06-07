import cv2
from .object_detection import Model
import numpy as np

import os
import io
import PIL.Image as Image
import torch

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        self.yolo = Model.YoloModel()
        self.model = self.yolo.load_model()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



        self.model = self.model.to(self.device)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        # image = torch.FloatTensor(image)
        # image = image.to(self.device)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        

        self.yolo.detect(image,self.model)


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