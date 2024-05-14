from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("update",views.update,name='update'),
    path("catalogo/file/<int:file_id>/",views.catalago_file,name="file"),
    path("catalogo/process/file/<int:file_id>/",views.process_file,name="process")
]