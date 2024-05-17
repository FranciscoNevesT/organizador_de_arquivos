from scipy.spatial import distance
from django.shortcuts import render, redirect
from ia.code.train import SimSiamTrainer
import torch
from torch.utils.data import DataLoader
from ia.code.model import BertModel,SimSiam,Classifier
from lightly import loss
from ia.code.dataset import FileText
import numpy as np
import torch.nn as nn

# Create your views here.
def train_model(request):
  if request.method == 'POST':
    backbone = BertModel()

    weights_path = "E:\organizador_de_arquivos\ia\weights"
    try:
      backbone.load_state_dict(torch.load("{}/backbone.pt".format(weights_path)))
    except:
      pass

    simsiam = SimSiam()

    try:
      simsiam.load_state_dict(torch.load("{}/simsiam.pt".format(weights_path)))
    except:
      pass

    data_path = "E:\organizador_de_arquivos\processed\\text"
    dataset = FileText(data_path=data_path, size=1000)


    classifier = Classifier(num_labels=dataset.__len_labels__())


    num_epoch = int(request.POST.get('num_epoch'))
    batch_size = int(request.POST.get('batch_size'))
    learning_rate = float(request.POST.get('learning_rate'))

    optimizer_backbone = torch.optim.Adam(backbone.parameters(),lr=learning_rate)
    optimizer_simsiam = torch.optim.Adam(simsiam.parameters(),lr=learning_rate)
    optimizer_classifier = torch.optim.Adam(classifier.parameters(),lr=learning_rate)

    criterion_simsiam = loss.NegativeCosineSimilarity()
    criterion_classfier = nn.BCELoss()


    train_dataloader = DataLoader(dataset, batch_size=batch_size)

    trainer = SimSiamTrainer(backbone=backbone,optimizer_backbone=optimizer_backbone,
                             simsiam = simsiam,optimizer_simsiam=optimizer_simsiam,criterion_simsiam=criterion_simsiam,
                             classifier=classifier,optimizer_classifier=optimizer_classifier,criterion_classfier=criterion_classfier,
                             train_dataloader=train_dataloader,val_dataloader=None)

    trainer.train(num_epochs=num_epoch)

    torch.save(backbone.state_dict(),"{}/backbone.pt".format(weights_path))
    torch.save(simsiam.state_dict(),"{}/simsiam.pt".format(weights_path))

    return redirect('/ia/results')  # Redirect to home page after training
  else:
    return render(request, 'ia/train_form.html')

def proximity(request):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  backbone = BertModel()

  weights_path = "E:\organizador_de_arquivos\ia\weights"

  try:
    backbone.load_state_dict(torch.load("{}/backbone.pt".format(weights_path)))
  except FileNotFoundError:
    return render(request, '404.html', status='404')

  simsiam = SimSiam()

  try:
    simsiam.load_state_dict(torch.load("{}/simsiam.pt".format(weights_path)))
  except FileNotFoundError:
    return render(request, '404.html', status='404')


  backbone.to(device)
  simsiam.to(device)

  data_path = "E:\organizador_de_arquivos\processed\\text"
  batch_size = 64

  dataset = FileText(data_path=data_path, size=1000,return_index=True)
  train_dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=False)

  num_files = len(dataset.text_index)

  X = []
  y = []
  with torch.no_grad():
    for (x0,x1,idx) in train_dataloader:
      x0 = x0.to(device)
      x1 = x1.to(device)

      r0 = backbone(x0)
      r1 = backbone(x1)

      _, p0 = simsiam(r0)
      _, p1 = simsiam(r1)

      p0 = p0.cpu().numpy()
      p1 = p1.cpu().numpy()

      for i in range(len(r0)):
        X.append(p0[i])
        y.append(idx[i].item())

        X.append(p1[i])
        y.append(idx[i].item())

  X = np.array(X)
  y = np.array(y)

  dist_matrix = np.zeros(shape=(num_files,num_files))
  comb_matrix = np.zeros(shape=dist_matrix.shape)

  for i in range(len(y)):
    for j in range(len(y)):
      idx0 = y[i]
      r0 = X[i]
      idx1 = y[j]
      r1 = X[j]

      dist = distance.cosine(r0,r1)

      comb_matrix[idx0,idx1] += 1
      dist_matrix[idx0,idx1] += dist

  dist_matrix = dist_matrix / comb_matrix

  context = {'data': []}
  for i in range(num_files):
    for j in range(num_files):
      context['data'].append({'x': i,
                              'y': j,
                              'value': dist_matrix[i,j]})

  return render(request, 'ia/proximity_results.html', context)