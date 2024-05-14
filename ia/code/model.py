import torch
import torch.nn as nn
from transformers import  BertForSequenceClassification
from lightly.models.modules import heads

class BertModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.backbone = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
    self.backbone.classifier = nn.Linear(768,512)


  def forward(self,x):
    x = self.backbone(x).logits
    return x


class SimSiam(nn.Module):
  def __init__(self, backbone):
    super().__init__()
    self.backbone = backbone
    self.projection_head = heads.SimSiamProjectionHead(512, 512, 128)
    self.prediction_head = heads.SimSiamPredictionHead(128, 64, 128)

  def forward(self, x):
    features = self.backbone(x).flatten(start_dim=1)
    z = self.projection_head(features)
    p = self.prediction_head(z)
    z = z.detach()
    return z, p




