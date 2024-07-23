#Depression-Prediction-Using-BiLSTM
#Dataset
The dataset used for this project is sourced from Kaggle. You can access the dataset thorugh this url: https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned
#Overview
This project aims to predict depression using a Bidirectional Long Short-Term Memory (BiLSTM) model trained on the given dataset. The dataset consists of cleaned Reddit posts related to depression, with labels indicating whether each post is associated with depression or not.  
#Code Description
The provided code implements the following steps:
Data Loading: Reads the dataset "depression_dataset_reddit_cleaned.csv" into a Pandas DataFrame.
Data Preprocessing: Performs data cleaning and preprocessing on the text data, including tokenization, lemmatization, and removing stopwords.
Model Building: Constructs a BiLSTM model using TensorFlow's Keras API, with an embedding layer, Bidirectional LSTM layer, and a dense output layer with sigmoid activation.
Model Training: Trains the BiLSTM model on the preprocessed text data, splitting the dataset into training and testing sets.
Model Evaluation: Evaluates the trained model on the testing set using accuracy, precision, recall, F1-score, and confusion matrix.
#Requirements
Python 3.x
TensorFlow
Pandas
Matplotlib
Seaborn
NLTK
#Usage
Clone this repository to your local machine.
Download the dataset from the provided Kaggle link and save it as "depression_dataset_reddit_cleaned.csv" in the repository folder.
Run the provided Python script.
#Results
The model performance metrics, including accuracy, precision, recall, F1-score, and confusion matrix, are printed to the console after model training and evaluation.
