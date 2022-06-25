# from object_detection.yolov5.hubconf import yolov5l
import torch
import os
import cv2


class YoloModel:

    def load_model(self):
        # self,model_name
        # model_name = os.path.join('object_detection/custom_models/',model_name)

        model = torch.hub.load('/yolov5','custom','yolov5s.pt',source='local',force_reload=True)
        # model = torch.hub.load('ultralytics/yolov5','yolov5l',force_reload=True )
        return model

 

    def detect(self,img,model):
        result = model(img)
        labels, cord = result.xyxyn[0][:,
                                    - 1].to('cpu').numpy(), result.xyxyn[0][:, :-1].to('cpu').numpy()
        # print(result.pandas().xyxy[0].xmin)

        n = len(labels)
        #print(n)
        x_shape, y_shape = img.shape[1], img.shape[0]


        for i in range(n):
            row = cord[i]
            l = int(labels[i])
            # lclasses.append(classes.iloc[l, :][0])
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]
                                                    * y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
    

            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 50, 30), 2)

            return (x1+x2)/2,(y1+y2)/2

        return 0,0
            # cv2.putText(img, str(classes.iloc[l, :][0]), (x1, y1-10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)









    