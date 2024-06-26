from django.urls import path
from . import views

from django.conf.urls.static import static
from django.conf import settings


# Specify paths : 
# > Connect a specific path to a view
urlpatterns = [
    path("experiments/",views.experiments, name="experiments"),
    path("doc/",views.documentation, name="documentation"),
    path("",views.home, name="home"),
    path("result/",views.experiments_list, name="experiments_list"),
    path("result/<int:experiment_id>/",views.result, name="result"),
    path("result/<int:experiment_id>/<int:image_id>/",views.image_result, name="image_result"),
    path("experiment_update/<int:experiment_id>/",views.get_experiment_update, name="experiment_update"),
    path("cred",views.cred, name="cred"),
]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

