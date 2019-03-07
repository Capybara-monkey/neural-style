from django.shortcuts import render, redirect
from .forms import PhotoForm
from .models import Photo
from django.http import HttpResponse
import glob
import os
import shutil

from .ns_model.generate import main


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


        # content画像を media/contentに保存
        photo = Photo()
        photo.image = form.cleaned_data["image"]
        photo.save()

        extension = glob.glob("./media/content/*")[0].split(".")[-1]
        content_path = "media/content/content." + extension
        result_path = "media/result/result.jpg"
        model_path = "transfer/ns_model/models/" + form["style"].data + ".model"

        main(content_path, model_path, result_path)

        return redirect("/transfer/result/")


def show_result(request):
    params = {
        "content_path": "/media/content/" + glob.glob("./media/content/*")[0].split("\\")[-1] ,
    }
    return render(request, "transfer/show_result.html", params)

def about(request):
    params = {}

    return render(request, "transfer/about.html", params)