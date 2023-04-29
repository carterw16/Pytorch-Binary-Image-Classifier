import generate_data, generate_model, train_model
import matplotlib.pyplot as plt

def plot_metrics(batches, val_losses, val_accs, save_path):
  # Create subplots for loss and accuracy
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

  # Plot validation loss
  ax1.plot(batches, val_losses)
  ax1.set_xlabel("Batch")
  ax1.set_ylabel("Validation Loss")
  ax1.set_title("Validation Loss over Batches")

  # Plot validation accuracy
  ax2.plot(batches, val_accs)
  ax2.set_xlabel("Batch")
  ax2.set_ylabel("Validation Accuracy")
  ax2.set_title("Validation Accuracy over Batches")

  # Save the plot to a file
  plt.savefig(save_path)
  plt.close()

def hyperparameter_search():
  """
  - define hyperparameters
    - lr, batch size
  - make grid for each hyperparameter
     - for lr in learning rates:
          for batch_size in bs:
              train for 1 epoch
  - save accuracy and loss
  - report best accuracy and loss
  grid search for hyperparameter tuning
  """
  
  lrs = [0.0025, 0.0035]
  batch_sizes = [16, 32]
  max_acc = 0
  best_hp = None
  time_limit = 5
  for lr in lrs:
    print("lr: ", lr)
    for bs in batch_sizes:
      print("batch size: ", bs)
      train_loader, val_loader = generate_data.get_data(batch_size=bs)
      model = generate_model.get_model()
      model, info = train_model.train_classifier(model, train_loader, val_loader, lr=lr, time_limit=time_limit)
      print("info: ", info)
      if info.get('max_acc') > max_acc:
        max_acc = info.get('max_acc')
        best_hp = {'bs': bs, 'lr': lr, 'max_acc': max_acc}
  print(best_hp)

if __name__ == '__main__':
  hyperparameter_search()
  
  
