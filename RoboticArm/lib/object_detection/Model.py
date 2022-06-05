import torch
import os


def load_model(model_name):
    model_name = os.path.join('custom_models/',model_name)

    model = torch.hub.load('yolov5','custom',path=model_name,source='local')

    return model







    