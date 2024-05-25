from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict, name='predict'),
    path('data_sample/', views.data_sample, name='data_sample')
]
