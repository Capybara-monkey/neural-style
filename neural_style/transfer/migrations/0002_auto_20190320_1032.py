# Generated by Django 2.0.3 on 2019-03-20 01:32

from django.db import migrations, models
import transfer.models


class Migration(migrations.Migration):

    dependencies = [
        ('transfer', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='photo',
            name='image',
            field=models.ImageField(upload_to=transfer.models.get_image_path),
        ),
    ]