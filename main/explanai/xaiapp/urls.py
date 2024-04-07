from django.urls import path
from . import views

# Specify paths : 
# > Connect a specific path to a view
urlpatterns = [
    path("experiments/",views.experiments, name="experiments"),
    path("",views.home, name="home"),
]

