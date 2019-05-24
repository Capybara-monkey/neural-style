from django import forms
import glob
import os

style_name = []
styles = glob.glob("./transfer/ns_model/*")
styles = os.listdir("./transfer/ns_model/pretrained_model")
for style in styles:
    style_name.append(style.split(".")[0])

class PhotoForm(forms.Form):
    STYLE_CHOICE = ((name, name) for name in style_name)

    style = forms.ChoiceField(
        label="スタイル",
        widget=forms.Select,
        choices=STYLE_CHOICE,
        required=True
    )

    image = forms.ImageField(label="画像")


class StyleForm(forms.Form):
    name = forms.CharField(max_length=20)
    image = forms.ImageField(label="style")
