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
    path("experiment_update/<int:experiment_id>/",views.get_experiment_update, name="experiment_update"),
]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

