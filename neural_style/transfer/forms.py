from django import forms

style_name = ['des_glaneuses', 'la_muse', 'mirror', 'udnie', 'wave_crop']

class PhotoForm(forms.Form):
    STYLE_CHOICE = ((name, name) for name in style_name)

    style = forms.ChoiceField(
        label="スタイル",
        widget=forms.Select,
        choices=STYLE_CHOICE,
        required=True
    )

    image = forms.ImageField()
