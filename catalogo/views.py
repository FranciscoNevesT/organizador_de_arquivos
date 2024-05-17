from django.shortcuts import render

from django.template import loader
from django.http import HttpResponse
from .models import File
import os
import time
from data_processing.read import ReadFiles


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
      File.objects.create(name=name, path=path, size=file_size,
                          type=file_type, creation_time=creation_time, modification_time=modification_time)
    except:
      entry = File.objects.filter(path=path)[0]

      entry.name = name
      entry.size = file_size
      entry.type = file_type
      entry.creation_time = creation_time
      entry.modification_time = modification_time

      entry.save()


def process_file(request, file_id):
  # Define the save directory (modify as needed)
  save_dir = "processed/text/"

  # Check if the save directory exists
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # Create the directory if it doesn't exist

  # Generate filename (consider using a more descriptive format)
  filename = f"processed_text_{file_id}.txt"

  if os.path.exists(os.path.join(save_dir, filename)):
    return index(request)

  readfiles = ReadFiles()

  file = File.objects.filter(numeric_id=file_id)

  if len(file) == 0:
    return render(request, '404.html', status=404)

  values = file[0]

  path = values.path
  type = values.type

  if type == '.pdf':
    text = readfiles.read_pdf(path)

  elif type == ".txt":
    text = readfiles.read_txt(path)
  else:
    return index(request)

  # Save the text to a file
  with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as text_file:
    text_file.write(text)

  # (Optional) Log or display a success message
  print(f"Text saved to: {os.path.join(save_dir, filename)}")

  return index(request)


def catalago_file(request, file_id):
  file = File.objects.filter(numeric_id=file_id)

  if len(file) == 0:
    return render(request, '404.html', status=404)

  values = file[0]

  path = values.path
  name = values.name
  size = values.size
  type = values.type
  creation_time = values.creation_time
  modification_time = values.modification_time

  persons = values.persons.all()
  themes = values.themes.all()

  template = loader.get_template("catalogo/file.html")

  values = {
    'file_id':file_id,
    'path': path,
    'name': name,
    'size': size,
    'type': type,
    'creation_time': creation_time,
    'modification_time': modification_time,
    'persons': persons,
    'themes': themes
  }

  return HttpResponse(template.render(values, request))


def update(request):
  update_dataset()

  return index(request)


def index(request):
  template = loader.get_template("catalogo/index.html")

  files = File.objects.all()

  values = {
    'files': files
  }

  return HttpResponse(template.render(values, request))
