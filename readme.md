# Handwritten-alphabet-recognizing-with-AI-


### Machine Learning Experiments and Model Comparison
This project implements various machine learning experiments for multi-class classification using a dataset of images. The focus is on data exploration, model training, and performance evaluation through a series of experiments. The results from different approaches are compared to identify the best-performing model.


### Experiments and Results

#### Experiment 1: Support Vector Machines (SVM) with Scikit-Learn
Train two SVM models:
  * Linear kernel
  * Non-linear kernel

Evaluation:
* Generate confusion matrices.
* Compute average F1 scores for the testing dataset.

#### Experiment 2: 
Logistic Regression (Implemented from Scratch)
Model Training: Implement one-versus-all logistic regression.
Performance Metrics:
Plot error and accuracy curves for both training and validation datasets.
Evaluate the model using a confusion matrix and average F1 scores on the testing dataset.

#### Experiment 3: Neural Networks with TensorFlow
Design Models: Create two neural networks with varying architectures (number of hidden layers, neurons, and activation functions).

Model Training: Train the networks and plot error and accuracy curves for training and validation datasets.

Model Saving: Save the best-performing model and reload it for testing.

Evaluation:
* Generate a confusion matrix and calculate average F1 scores for the testing dataset.
* Test the best model with images representing alphabetical letters from the team members’ names.


### Dataset
The dataset used for this project is the A-Z Handwritten Alphabets in .csv format, available at Kaggle: https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format
