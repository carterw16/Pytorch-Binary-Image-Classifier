import torch
from torchvision import datasets, transforms

class MyDataset():
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def image_transforms():
  # Define the transformations to be applied to each image
  train_transform = transforms.Compose([
      transforms.RandomRotation(30),
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
  ])
  val_transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  return train_transform, val_transform

def get_data(path='Data/Train_Data', batch_size=32):
  # Load the dataset using PyTorch's ImageFolder class
  dataset = datasets.ImageFolder(path, transform=None)

  # Split the dataset into training and validation sets using random_split
  train_size = int(0.8 * len(dataset))
  val_size = len(dataset) - train_size
  train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

  train_transform, val_transform = image_transforms()
  train_dataset = MyDataset(train_dataset, transform=train_transform);
  val_dataset = MyDataset(val_dataset, transform=val_transform);
  
  # Define the dataloaders for training and validation sets
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  return train_loader, val_loader
