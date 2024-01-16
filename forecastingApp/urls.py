from django.urls import path
from . import views

urlpatterns = [
    path("prediccion/", views.Prediccion.as_view()),
]