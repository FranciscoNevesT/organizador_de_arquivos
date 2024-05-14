import torch
from torch import nn
from torch.utils.data import DataLoader
from model import BertModel,SimSiam
from lightly import loss
from dataset import FileText

class SimSiamTrainer:
  def __init__(self, model, optimizer, criterion, train_dataloader, val_dataloader=None):
    self.model = model
    self.optimizer = optimizer
    self.criterion = criterion
    self.train_dataloader = train_dataloader
    self.val_dataloader = val_dataloader
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)

  def train_epoch(self):
    self.model.train()  # Set model to training mode

    total_loss = 0
    for [x0,x1] in self.train_dataloader:
      x0 = x0.to(self.device)
      x1 = x1.to(self.device)

      # Forward pass, compute loss
      z0, p0 = model(x0)
      z1, p1 = model(x1)

      loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))

      # Backward pass and optimize
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      total_loss += loss.item()

    return total_loss / len(self.train_dataloader)

  def validate_epoch(self):
    if not self.val_dataloader:
      return None

    self.model.eval()  # Set model to evaluation mode

    with torch.no_grad():
      total_loss = 0
      for data in self.val_dataloader:
        inputs, targets = data
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        z1, z2 = self.model(inputs)
        loss = self.criterion(z1, z2.detach())

        total_loss += loss.item()

    return total_loss / len(self.val_dataloader)

  def train(self, num_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
      print(f"Epoch {epoch + 1}/{num_epochs}")

      train_loss = self.train_epoch()
      train_losses.append(train_loss)

      if self.val_dataloader:
        val_loss = self.validate_epoch()
        val_losses.append(val_loss)
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
      else:
        print(f"Train Loss: {train_loss:.4f}")

    return train_losses, val_losses

backbone = BertModel()
model = SimSiam(backbone)
optimizer = torch.optim.Adam(model.parameters())
criterion = loss.NegativeCosineSimilarity()


data_path = "E:\organizador_de_arquivos\processed\\text"

dataset = FileText(data_path=data_path,size=1000)
train_dataloader = DataLoader(dataset,batch_size=64)

trainer = SimSiamTrainer(model=model,optimizer=optimizer,criterion=criterion,train_dataloader=train_dataloader,val_dataloader=None)
print(trainer.train(100))