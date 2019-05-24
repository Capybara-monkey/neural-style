from django.shortcuts import render, redirect
from django.views.generic import TemplateView
import glob
import os
import shutil
from PIL import Image

from .forms import PhotoForm, StyleForm
from .models import Photo, Style
from .tasks import learn_style

from .ns_model.model import NS_MODEL
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np
import tensorflow as tf


graph = tf.Graph()
with graph.as_default():
    session = tf.Session()
    with session.as_default():
        model = NS_MODEL()

class AboutView(TemplateView):
    template_name = "transfer/about.html"


class ResultView(TemplateView):
    template_name = "transfer/show_result.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["content_path"] = "/media/content/" + glob.glob("./media/content/*")[0].split("\\")[-1]
        # media/resultが空かどうかを確認
        if os.listdir('./media/result'):
            context["done"] = True
        else:
            context["done"] = False
        return context


def home(request):
    #スタイル画像のファイル名とurlの取得
    style_name = [r.split("\\")[-1].split(".")[0] for r in glob.glob("./transfer/static/style/*")]
    style_url = ["style_with_title/" + (r.split("\\")[-1]) for r in glob.glob("./transfer/static/style_with_title/*")]

    params = {
    "form": PhotoForm(),
    "style_name": style_name,
    "style_url": style_url,
    }
    if request.method=="GET":
        #content, result ディレクトリの初期化
        shutil.rmtree("./media/content")
        shutil.rmtree("./media/result")
        os.mkdir("./media/content")
        os.mkdir("./media/result")
        return render(request, 'transfer/home.html', params)

    elif request.method=="POST":
        form = PhotoForm(request.POST, request.FILES)
        if not form.is_valid():
            raise ValueError('invalid form')

        # content画像を media/content に保存
        photo = Photo()
        photo.image = form.cleaned_data["image"]
        photo.save()

        extension = glob.glob("./media/content/*")[0].split(".")[-1]
        content_path = "media/content/content." + extension
        result_path = "media/result/result.jpg"
        style = form.data["style"]

        # 画像サイズが大きすぎる場合には縮小
        img = Image.open(content_path)
        width, height = img.size
        if width > 1000:
            resize_width=1000
            resize_height = height * (resize_width/width)
            img = img.resize((int(resize_width), int(resize_height)))
            img.save(content_path)

        weihts_name = "transfer/ns_model/pretrained_model/"+ style + ".h5"
        global graph, session
        K.set_session(session)
        with graph.as_default():
            model.model.load_weights(weihts_name)

        K.set_session(session)
        with graph.as_default():
            img = img_to_array(load_img(content_path, target_size=(224, 224)))
            trans_img = model.model_gen.predict(np.expand_dims(img, axis=0))
        array_to_img(trans_img[0]).save(result_path)
        return redirect("/transfer/result/")

def learn(request):
    if request.method == "GET":
        form = StyleForm()
        params = {"form": form}
        return render(request, 'transfer/learn.html', params)

    if request.method == "POST":
        form = StyleForm(request.POST, request.FILES)
        if not form.is_valid():
            raise ValueError("invalid form")

        style = form.data["name"]

        # content画像を media/content に保存
        img = Style()
        img.image = form.cleaned_data["image"]
        img.save()

        learn_style.delay(style=style)
        return redirect("/transfer/")






