# Cat vs Dog Binary Image Classifier with Pytorch
by Carter Weaver

## Description
This is a simple example of a convolutional neural network trained with Pytorch to classify images as cats or dogs. The dataset used can be found in Data/Train_Data.

## Hyperparameter Search
`python training_tools.py`

The hyperparameter_search() function performs a grid search for hyperparameter tuning. It loops through a selection of hyperparameter values (batch_size and learning rate) and trains the model for a short amount of time, returning the hyperparameter combination that gives the best accuracy. For the purposes of this experiment, the set of hyperparameter options is kept small but can be expanded to include other values.

## Training the Model
`python train_model.py`

The train_model() function first gets the data and preprocesses it using generate_data.py. It then generates the model. This project uses densenet121, a pretrained model for image classification available through Pytorch, and simply retrains the classifier to be binary. This makes for an accurate model since it's already trained for images and saves times since only part of it requires training. 

The model is then trained on a training set of images. Every 5 batches of data the model is tested against a validation set of images and the results from that test are saved to batch_results.pth (loss and accuracy). If a new best accuracy was achieved, the model's state is saved to model.pth. After training is complete, the data from batch_results.pth is used to plot validation loss and accuracy over time in plots.png.

<img src="plots.png?raw=true" width="600">

## Making a prediction
`python predict.py <path to image>`

The predict_image() function preprocesses a given image, loads the saved model from model.pth, and uses it to classify the image as a cat or a dog.

### Example:
Input Image: 
<img src="example_dog.jpg?raw=true" width="400">

Command-Line Output: Dog
