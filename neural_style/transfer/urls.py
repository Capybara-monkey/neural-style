from django.urls import path
from transfer import views

urlpatterns = [
    path('', views.home, name='home'),
    path('result/', views.ResultView.as_view(), name="result"),
    path('about/', views.AboutView.as_view(), name='about'),
]