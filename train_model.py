import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import generate_model, generate_data, training_tools
from torch.autograd import Variable
import time
import pickle


def train_classifier(model, train_loader, val_loader, lr=0.0035, epochs=1, time_limit=None):
  # Use GPU if it's available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # Only train the classifier parameters, feature parameters are frozen
  optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
  criterion = nn.NLLLoss()

  model.to(device);
  steps = 0
  running_loss = 0
  print_every = 5
  max_acc = 0
  start_time = time.time()
  batch_results = {'batches': [], 'loss': [], 'accuracy': []}
  model.train()
  for epoch in range(epochs):
    print(len(iter(train_loader)))
    for inputs, labels in iter(train_loader):
      steps += 1
      # Move input and label tensors to the default device
      inputs, labels = inputs.to(device), labels.to(device)
      
      optimizer.zero_grad()
      
      logps = model.forward(inputs)
      loss = criterion(logps, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      print(steps, running_loss);
    
      if (steps % print_every == 0):
        val_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
          
          for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            
            val_loss += batch_loss.item()
            
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
              
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/print_every:.3f}.. "
              f"Validation loss: {val_loss/len(val_loader):.3f}.. "
              f"Validation accuracy: {accuracy/len(val_loader):.3f}")
        
        batch_results.get('batches').append(steps)
        batch_results.get('loss').append(val_loss/len(val_loader))
        batch_results.get('accuracy').append(accuracy/len(val_loader))
        with open("batch_results.pth", "wb") as f:
          pickle.dump(batch_results, f)
        
        if ((accuracy / len(val_loader)) > max_acc):
          max_acc = (accuracy / len(val_loader))
          print("new best accuracy")
          torch.save({
            'model_name': 'cats_vs_dogs',
            'model_state_dict': model.state_dict(),
            'epochs': epochs,
            'train_loss': running_loss/print_every,
            'val_loss': val_loss/len(val_loader),
            'val_accuracy': accuracy/len(val_loader)
            },
            "model.pth"
          )
      
        minutes_elapsed = (time.time()-start_time)/60
        if time_limit is not None and minutes_elapsed > time_limit:
          return model, {'max_acc': max_acc}  
        running_loss = 0
        model.train()
  
  return model, {'max_acc': max_acc}


def train_model():
  train_loader, val_loader = generate_data.get_data()
  model = generate_model.get_model()
  model = train_classifier(model, train_loader, val_loader, epochs=3)
  
  # read pickle and graph loss and acc
  with open("batch_results.pth", "rb") as f:
    results = pickle.load(f)
  training_tools.plot_metrics(results['batches'], results['loss'], results['accuracy'], 'plots.png')
  
  return model

if __name__ == '__main__':
  model = train_model()
