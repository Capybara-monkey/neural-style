from celery import shared_task

from .ns_model.transform_net import TransformNet

Net = TransformNet()


@shared_task
def predict(style, result_path, content_path):
    Net.predict(style=style, output_file=result_path, input_file=content_path)