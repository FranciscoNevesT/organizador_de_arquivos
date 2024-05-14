import torch
from torch.utils.data import Dataset
import os
import nltk
import numpy as np


class FileText(Dataset):
  def __init__(self, data_path, sequence_length = 32, size = None, transform=None):
    """
    Args:
        data_path (str): Path to the data file (e.g., CSV, image directory)
        transform (callable, optional): Function to apply transformations to the data. Defaults to None.
    """
    self.transform = transform
    self._load_data(data_path)  # Replace this with your data loading logic
    self.sequence_length = sequence_length

    if size is None:
      self.size = len(self.text_index)
    else:
      self.size = size

  def _load_data(self, data_path):
    """
    This function should be implemented to load your specific data format (e.g., read CSV, load images)
    and return a list of data samples. Each sample can be a dictionary, tuple, or any custom object
    containing your data points.
    """
    texts = []
    for file_path in os.listdir(data_path):
      file_path = "{}/{}".format(data_path, file_path)

      with open(file_path, "r", encoding="utf8") as file:
        texts.append(file.read())

    text_all = " ".join(texts)
    tokens = set(text_all.split())

    vocab = {token: i for i, token in enumerate(tokens)}
    text_index = [[vocab[word] for word in text.split()] for text in texts]

    self.vocab = vocab
    self.text_index = text_index

  def __len__(self):
    return self.size

  def __getitem__(self, idx):
    idx = idx % len(self.text_index)

    x0 = np.random.randint(0,len(self.text_index[idx]) - self.sequence_length)
    x0 = self.text_index[idx][x0:x0+self.sequence_length]

    x1 = np.random.randint(0, len(self.text_index[idx]) - self.sequence_length)
    x1 = self.text_index[idx][x1:x1 + self.sequence_length]

    x0 = torch.tensor(x0)
    x1 = torch.tensor(x1)

    return x0,x1


