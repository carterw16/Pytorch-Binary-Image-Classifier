import torch
from torchvision import transforms
from PIL import Image
import generate_model, generate_data

def predict_image(image_path):

  train_transform, val_transform = generate_data.image_transforms()
  image = Image.open(image_path)

  # Apply the transformation to the image
  image_tensor = val_transform(image).float()
  image_tensor = image_tensor.unsqueeze_(0)

  # Load the saved model
  checkpoint = torch.load('model.pth')
  model = generate_model.get_model()
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()

  # Make prediction using the model
  with torch.no_grad():
      output = model(image_tensor)
      _, predicted = torch.max(output, 1)

  return "Cat" if predicted.item() == 0 else "Dog"

if __name__ == '__main__':
  import sys
  result = predict_image(sys.argv[1])
  print(result)