import torch

class SimSiamTrainer:
  def __init__(self, backbone,optimizer_backbone,
              simsiam,optimizer_simsiam,criterion_simsiam,
               classifier,optimizer_classifier,criterion_classfier,


               train_dataloader, val_dataloader=None):

    self.backbone = backbone
    self.optimizer_backbone = optimizer_backbone

    self.simsiam = simsiam
    self.optimizer_simsiam = optimizer_simsiam
    self.criterion_simsiam = criterion_simsiam

    self.classifier = classifier
    self.optimizer_classifier = optimizer_classifier
    self.criterion_classifier = criterion_classfier


    self.train_dataloader = train_dataloader
    self.val_dataloader = val_dataloader
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.backbone.to(self.device)
    self.simsiam.to(self.device)
    self.classifier.to(self.device)

  def train_epoch(self):
    """
    Trains the model for one epoch.

    Returns:
        avg_loss_simsiam: Average loss of SimSiam over the epoch.
        avg_loss_classifier: Average loss of the classifier over the epoch.
    """
    self.backbone.train()
    self.simsiam.train()
    self.classifier.train()

    total_loss_simsiam = 0
    total_loss_classifier = 0

    for [x0, x1, labels] in self.train_dataloader:
      # Move data to device
      x0 = x0.to(self.device)
      x1 = x1.to(self.device)
      labels = labels.to(self.device).float()

      data = [x0, x1, labels]

      # Extract features
      features_x0, features_x1 = self._extract_features(data)

      z0, p0 = self.simsiam(features_x0)
      z1, p1 = self.simsiam(features_x1)

      loss_simsiam = 0.5 * (self.criterion_simsiam(z0, p1) + self.criterion_simsiam(z1, p0))

      self.optimizer_backbone.zero_grad()
      self.optimizer_simsiam.zero_grad()

      loss_simsiam.backward()

      self.optimizer_backbone.step()
      self.optimizer_simsiam.step()


      features_x0, features_x1 = self._extract_features(data)

      labels0 = self.classifier(features_x0)
      labels1 = self.classifier(features_x1)

      loss_classifier = 0.5 * (self.criterion_classifier(labels0, labels) + self.criterion_classifier(labels1, labels))

      self.optimizer_backbone.zero_grad()
      self.optimizer_classifier.zero_grad()

      loss_classifier.backward()

      self.optimizer_backbone.step()
      self.optimizer_classifier.step()


      # Update total losses
      total_loss_simsiam += loss_simsiam.item()
      total_loss_classifier += loss_classifier.item()

    avg_loss_simsiam = total_loss_simsiam / len(self.train_dataloader)
    avg_loss_classifier = total_loss_classifier / len(self.train_dataloader)

    return avg_loss_simsiam, avg_loss_classifier

  # Helper functions for clarity
  def _extract_features(self, data):
    x0, x1, _ = data
    features_x0 = self.backbone(x0).flatten(start_dim=1)
    features_x1 = self.backbone(x1).flatten(start_dim=1)
    return features_x0, features_x1

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

      train_loss_simsiam,train_loss_classifier = self.train_epoch()

      if self.val_dataloader:
        val_loss = self.validate_epoch()
        val_losses.append(val_loss)
        print(f"Train Loss: {train_loss_simsiam:.4f}, Validation Loss: {val_loss:.4f}")
      else:
        print(f"Train Loss: Simsiam {train_loss_simsiam:.4f} | Classifer {train_loss_classifier:.4f}")

    return train_losses, val_losses




