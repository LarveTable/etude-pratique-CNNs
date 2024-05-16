from django.urls import path
from . import views

from django.conf.urls.static import static
from django.conf import settings


# Specify paths : 
# > Connect a specific path to a view
urlpatterns = [
    path("experiments/",views.experiments, name="experiments"),
    path("",views.home, name="home"),
    path("result/<int:experiment_id>/",views.result, name="result"),
]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

