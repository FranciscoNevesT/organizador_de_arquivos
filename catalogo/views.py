from django.shortcuts import render

from django.template import loader
from django.http import HttpResponse
from .models import File
import os
import time


def update_dataset():
  for i in os.listdir("files"):
    file_path = "files/{}".format(i)

    name = os.path.splitext(file_path)[0].split("/")[1]
    path = os.path.abspath(file_path)
    file_size = os.path.getsize(file_path)
    file_type = os.path.splitext(file_path)[1]

    file_creation_time = os.path.getctime(file_path)
    file_modification_time = os.path.getmtime(file_path)

    # Convert creation and modification time to readable format
    creation_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_creation_time))
    modification_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_modification_time))

    try:
      File.objects.create(name=name,path=path,size = file_size,
                          type = file_type, creation_time = creation_time,modification_time =modification_time)
    except:
      entry = File.objects.filter(path = path)[0]

      entry.name = name
      entry.size = file_size
      entry.type = file_type
      entry.creation_time = creation_time
      entry.modification_time = modification_time

      entry.save()
      print(entry)
      pass


def index(request):
  update_dataset()

  template = loader.get_template("catalogo/index.html")

  files = File.objects.all()


  values = {
      'files':files
  }

  return HttpResponse(template.render(values, request))
