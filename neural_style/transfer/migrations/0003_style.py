# Generated by Django 2.0.3 on 2019-05-24 14:24

from django.db import migrations, models
import transfer.models


class Migration(migrations.Migration):

    dependencies = [
        ('transfer', '0002_auto_20190320_1032'),
    ]

    operations = [
        migrations.CreateModel(
            name='Style',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to=transfer.models.get_style_path)),
            ],
        ),
    ]
