from django.shortcuts import render, redirect
from keras import backend as K
from django.http import HttpResponse
from django.views.generic import TemplateView
import glob
import os
import shutil
from PIL import Image

from .forms import PhotoForm
from .models import Photo
from .tasks import predict

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


        K.clear_session()
        predict.delay(style, result_path, content_path)
        return redirect("/transfer/result/")



class AboutView(TemplateView):
    template_name = "transfer/about.html"


class ResultView(TemplateView):
    template_name = "transfer/show_result.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["content_path"] = "/media/content/" + glob.glob("./media/content/*")[0].split("\\")[-1]

        return context
