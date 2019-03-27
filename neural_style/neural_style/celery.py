import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'neural_style.settings')

app = Celery('neural_style')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
