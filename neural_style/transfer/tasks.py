from celery import shared_task
from .ns_model.model import get_model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np
import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    session = tf.Session()
    with session.as_default():
        Net, model = get_model()

@shared_task
def predict(style, result_path, content_path):
    weihts_name = "transfer/ns_model/pretrained_model/"+ style + ".h5"
    global graph, session
    K.set_session(session)
    with graph.as_default():
        model.load_weights(weihts_name)

    K.set_session(session)
    with graph.as_default():
        img = img_to_array(load_img(content_path, target_size=(224, 224)))
        trans_img = Net.predict(np.expand_dims(img, axis=0))
    array_to_img(trans_img[0]).save(result_path)


