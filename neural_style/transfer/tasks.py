from celery import shared_task
from .ns_model.transform_net import TransformNet
from django.shortcuts import redirect
from keras import backend as K
import os

Net = TransformNet()

@shared_task
def predict(style, result_path, content_path):
    K.clear_session()
    Net.predict(style=style, output_file=result_path, input_file=content_path)



