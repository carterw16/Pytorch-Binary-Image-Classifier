import torch
from torch import nn
from torchvision import models
from collections import OrderedDict

def get_model():
  # Load the pre-trained densenet model and modify the last layer for binary classification
  model = models.densenet121(weights='DEFAULT')

  # Freeze parameters so we don't backprop through them
  for param in model.parameters():
      param.requires_grad = False

  classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(1024, 500)),
                            ('relu', nn.ReLU()),
                            ('fc2', nn.Linear(500, 2)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
      
  model.classifier = classifier

  return model
  