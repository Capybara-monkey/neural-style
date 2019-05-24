from celery import shared_task
from .ns_model.model import NS_MODEL
from tensorflow.python.keras import backend as K
import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    session = tf.Session()
    with session.as_default():
        model = NS_MODEL()

@shared_task
def learn_style(style):
    global graph, session
    K.set_session(session)
    with graph.as_default():
        model.learn(style)


