import torch
import torch.nn as nn
from transformers import  BertForSequenceClassification
from lightly.models.modules import heads

class BertModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.backbone = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
    self.backbone.classifier = nn.Linear(768,512)
    self.backbone.bert.embeddings.word_embeddings = nn.Embedding(100000,768,padding_idx=0)


  def forward(self,x):
    x = self.backbone(x).logits
    return x


class SimSiam(nn.Module):
  def __init__(self):
    super().__init__()
    self.projection_head = heads.SimSiamProjectionHead(512, 512, 128)
    self.prediction_head = heads.SimSiamPredictionHead(128, 64, 128)

  def forward(self, features):
    z = self.projection_head(features)
    p = self.prediction_head(z)
    z = z.detach()
    return z, p


class LinearBlock(nn.Module):
  def __init__(self,in_features,out_features):
    super().__init__()

    self.linear = nn.Sequential(nn.Linear(in_features=in_features,out_features=out_features),
                                nn.Dropout(0.2),
                                nn.BatchNorm1d(out_features),
                                nn.ReLU(inplace=True),)

  def forward(self,x):
    return self.linear(x)

class Classifier(nn.Module):
  def __init__(self, num_labels, in_features = 512):
    super().__init__()

    self.linear = nn.Sequential( LinearBlock(in_features=in_features,out_features=in_features),
                                 LinearBlock(in_features=in_features,out_features=in_features),
                                 LinearBlock(in_features=in_features,out_features=in_features),)

    self.out = nn.Sequential(nn.Linear(in_features=in_features,out_features=num_labels),
                             nn.Sigmoid())

  def forward(self, features):
    logits = self.linear(features)
    logits = self.out(logits)

    return logits

