from django.db import models


def get_image_path(instance, filename):
    return "content/content.%s" % filename.split(".")[-1]

def get_style_path(instance, filename):
    return "style/%s" % filename

class Photo(models.Model):
    image = models.ImageField(upload_to=get_image_path)

class Style(models.Model):
    image = models.ImageField(upload_to=get_style_path)
