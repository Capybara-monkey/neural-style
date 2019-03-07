from django.urls import path
from transfer import views

urlpatterns = [
    path('', views.home, name='home'),
    path('result/', views.show_result, name="result"),
    path('about/', views.about, name='about'),
]