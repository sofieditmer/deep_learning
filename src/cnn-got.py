#!/usr/bin/env python
"""
Info: This script trains a deep learning CNN model on the dialogue from all 8 Game of Thrones seasons to learn to classify lines according to season.

Parameters:
    (optional) input_file: str <path-to-training-data>, default = "game_of_thrones_script.csv"
    (optional) chunk_size: int <size-of-chunks>, default = 2
    (optional) n_epochs: int <number-of-epochs>, default = 10
    (optional) batch_size: int <size-of-batches>, default = 10
    (optional) test_size: float <size-of-test-split>, default = 0.40
    (optional) embedding_dim: int <number-of-embedding-dimensions>, default = 100
    (optional) l2_value: float <regularization-value>, default = 0.001
    (optional) dropout_value: float <likelihood-of-dropping-nodes>, default = 0.01
    (optional) optimizer: str <choice-of-optimizer-algorithm>, default = "adam"
    (optional) n_words: int <number-of-words-to-initalize-tokenizer>, default = 5000

Usage:
    $ python cnn-got.py
    
Output:
    - cnn_model_summary.txt: a summary of the CNN model architecture.
    - cnn_classification_metrics.txt: classificaiton report of the model.
    - cnn_model_history.png: loss/accuracy history of model. 
"""

### DEPENDENCIES ###

# Core libraries
import os
import sys
sys.path.append(os.path.join(".."))

# contextlib
from contextlib import redirect_stdout # for saving output

# import utility functions
import utils.preprocessing_utils as preprocess
import utils.utils as utils

# Machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, Flatten, GlobalMaxPool1D, Conv1D, Dropout)
from tensorflow.keras.optimizers import SGD, Adam # optimization algorithms
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2 # regularization

# matplotlib
import matplotlib.pyplot as plt

# argparse
import argparse

### MAIN FUNCTION ###

def main():
    
    ### ARGPARSE ###
    
    # Initialize ArgumentParser class
    ap = argparse.ArgumentParser()
    
    # Argument 1: Path to training data
    ap.add_argument("-i", "--input_file",
                    type = str,
                    required = False, # this argument is not required
                    help = "Name of data",
                    default = "game_of_thrones_script.csv") # default data
    
    # Argument 2: Chunk size 
    ap.add_argument("-c", "--chunk_size",
                    type = int,
                    required = False, # this argument is not required
                    help = "Define the size of the chunks, i.e. how many sentences should be grouped together.",
                    default = 2) # default chunk size
    
    # Argument 3: Number of epochs
    ap.add_argument("-e", "--n_epochs",
                    type = int,
                    required = False, # this argument is not required
                    help = "The number of epochs",
                    default = 10) # default number of epochs
    
    # Argument 4: Size of test split
    ap.add_argument("-ts", "--test_split",
                    type = float,
                    required = False, # this argument is not required
                    help = "The size of the test split defined as float, e.g. 0.25",
                    default = 0.25) # default size of test is 25%
    
    # Argument 5: Number of embedding dimensions
    ap.add_argument("-em", "--embedding_dim",
                    type = int,
                    required = False, # this argument is not required
                    help = "The number of embedding dimensions",
                    default = 100) # default embedding dimensions
    
    # Argument 6: Regularization value. 
    ap.add_argument("-r", "--regularization_value",
                    type = float,
                    required = False, # this argument is not required
                    help = "The regularization value. The smaller the more regularization",
                    default = 0.001) # default l2-value.
    
    # Argument 7: Dropout value
    ap.add_argument("-d", "--dropout_value",
                    type = float,
                    required = False, # this argument is not required
                    help = "The dropout value which specifies how much to drop.",
                    default = 0.01) # default dropout value
    
    # Argument 8: Optimizer 
    ap.add_argument("-o", "--optimizer",
                    type = str,
                    required = False, # this argument is not required
                    help = "Choose the optimizer algorithm you want. You can either choose 'SGD' or 'adam'.",
                    default = 'adam') # default optimizer

    
    # Argument 9: Number of words to intialize tokenizer
    ap.add_argument("-n", "--n_words",
                    type = int,
                    required = False, # this argument is not required
                    help = "Number of words to initialize the tokenizer",
                    default = 5000) # default number of words
    
    # Parse arguments
    args = vars(ap.parse_args())
    
    # Save input parameters
    input_file = os.path.join("..", "data", args["input_file"])
    chunk_size = args["chunk_size"]
    n_epochs = args["n_epochs"]
    test_size = args["test_split"]
    embedding_dim = args["embedding_dim"]
    l2_value = args["regularization_value"]
    dropout_value = args["dropout_value"]
    optimizer = args["optimizer"]
    n_words = args["n_words"]
        
    # Create output directory
    if not os.path.exists(os.path.join("..", "output")):
        os.mkdir(os.path.join("..", "output"))

    # User message
    print("\n[INFO] Initializing the construction of the CNN model...")
    
    # Load and preprocess data
    print(f"\n[INFO] Loading, preprocessing, and chunking '{input_file}'...")
    preprocessed_data = preprocess.load_and_tokenize(input_file)
    preprocessed_df = preprocess.chunk_data(preprocessed_data, chunk_size)
    
    # Instantiate CNN_classifier class
    cnn = CNN_classifier(preprocessed_df)
    
    # Create train-test split
    print(f"\n[INFO] Creating train-test split with test size of {test_size}...")
    X_train, X_test, y_train, y_test, sentences, labels = cnn.create_train_test(test_size)
    
    # Binarize labels
    print("\n[INFO] Binarizing labels...")
    y_train_binarized, y_test_binarized = cnn.binarize(y_train, y_test)
    
    # Create word embeddings
    print("\n[INFO] Creating word embeddings...")
    tokenizer, vocab_size, X_train_tokens, X_test_tokens = cnn.create_word_embeddings(X_train, X_test, n_words)
    
    # Add paddings to ensure equal length
    print("\n[INFO] Adding padding to ensure equal document length...")
    maxlen, X_train_pad, X_test_pad = cnn.pad(X_train_tokens, X_test_tokens)
    
    # Define CNN model architecture
    print("\n[INFO] Defining CNN model architecture...")
    model = cnn.define_cnn_model(vocab_size, l2_value, embedding_dim, dropout_value, optimizer, maxlen)
    
    # Train and test the model
    print("\n[INFO] Training model...")
    cnn.train_model(model, X_train_pad, y_train_binarized, X_test_pad, y_test_binarized, n_epochs)
    
    # Evalute model
    print("\n[INFO] Evaluating model...")
    classification_metrics = cnn.evaluate_model(model, X_test_pad, X_train_pad, y_train_binarized, y_test_binarized, labels, n_epochs)
    print(f"\n [INFO] Below are the classification metrics for the trained model. \n {classification_metrics} \n")
    
    # Plot model history
    print("\n[INFO] Plotting accuracy/loss history and saving to 'output' directory...")
    cnn.plot_history(history, n_epochs)
    
    # User message
    print("\n[INFO] Done! You have now defined and trained a deep learning CNN classifier model on the Game of Thrones dialogue from all 8 seasons. The results can be found in the output directory.\n")
    
    
### CNN CLASSIFIER ###
    
# Creating CNN classifier class 
class CNN_classifier:
    
    # Intialize CNN classifier class
    def __init__(self, preprocessed_df):
        
        # Receive input
        self.preprocessed_df = preprocessed_df
        
        
    def create_train_test(self, test_size):
        """
        This method creates X_train, X_test, y_train, and y_test based on the preprocessed chunked dialogue.
        """
        # Extract seasons as labels
        labels = self.preprocessed_df['Season'].values
        
        # Create train data
        sentences = self.preprocessed_df["Chunks"].values

        # Create training and test split using sklearn
        X_train, X_test, y_train, y_test = train_test_split(sentences,
                                                            labels,
                                                            test_size=test_size, # default is 0.25
                                                            random_state=42) # random state for reproducibility
    
        return X_train, X_test, y_train, y_test, sentences, labels


    def binarize(self, y_train, y_test):
        """
        This method binarizes the training and test labels.
        """
        # Initialize binarizer
        lb = LabelBinarizer()
    
        # Binarize training labels
        y_train_binarized = lb.fit_transform(y_train)
    
        # Binarize test labels
        y_test_binarized = lb.fit_transform(y_test)
        
        return y_train_binarized, y_test_binarized
    
    
    def create_word_embeddings(self, X_train, X_test, n_words):
        """
        This method creates word embeddings using tf.keras.Tokenizer() which allows for efficiently converting text to numbers.
        """
        # Initialize tokenizer
        tokenizer = Tokenizer(num_words=n_words)
    
        # Fit tokenizer to training data
        tokenizer.fit_on_texts(X_train)
    
        # Use the tokenizer to create sequences of tokens for both training and test data
        X_train_tokens = tokenizer.texts_to_sequences(X_train)
        X_test_tokens = tokenizer.texts_to_sequences(X_test)
    
        # Define overall vocabulary size
        vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    
        return tokenizer, vocab_size, X_train_tokens, X_test_tokens


    def pad(self, X_train_tokens, X_test_tokens):
        """
        This methods pads the tokenized documents to make sure that they are of equal length. 
        Rather than setting an arbitrary maximum length, padding is done by first computing the maximum length of all 
        documents and then adding 0s to all documents to match the maximum length. 
        Computing the maxlen rather than setting an arbitrary value ensures that we actually consider the data in question. 
        """
        # Define max length for a doc by finding the maximum length of both the training tokens and the test tokens
        maxlen_train = len(max(X_train_tokens, key=len))
        maxlen_test = len(max(X_test_tokens, key=len))
        maxlen = max(maxlen_train, maxlen_test)

        # Pad training data to max length
        X_train_pad = pad_sequences(X_train_tokens,
                                    padding='post', # post = puts the pads at end of the sequence. Sequences can be padded "pre" or "post"
                                    maxlen=maxlen)
    
        # Pad testing data to max length
        X_test_pad = pad_sequences(X_test_tokens,
                                   padding='post',
                                   maxlen=maxlen)
    
        return maxlen, X_train_pad, X_test_pad


    def define_cnn_model(self, vocab_size, l2_value, embedding_dim, dropout_value, optimizer, maxlen):
        """
        This method defines CNN model architecture.
        """
        # Clearing sessions (deleting already trained models)
        tf.keras.backend.clear_session()

        # Define L2 regularizer. The smaller the value, the more regularization. Default is 0.0001
        l2 = L2(l2_value)
    
        # Intialize sequential model
        model = Sequential()
    
        # Add embedding layer that converts the numerical representations of the sentences into a dense, embedded representation
        model.add(Embedding(input_dim = vocab_size,
                            output_dim = embedding_dim, # default is 100
                            input_length = maxlen)) # based on the data    
    
        # Add convolutional layer with input of 128 and 1D kernel of 5
        model.add(Conv1D(128, 5,
                         activation = 'relu', # ReLU activation function
                         kernel_regularizer = l2)) # L2 regularization 
    
        # Global max pooling
        model.add(GlobalMaxPool1D())
    
        # Add dropout layer to reduce overfitting
        model.add(Dropout(dropout_value)) # default is 0.01, i.e. 1%
    
        # Add dense layer with L2-regularizer to reduce overfitting
        model.add(Dense(32, activation = 'relu',
                        kernel_regularizer = l2))
    
        # Add another dropout layer to reduce overfitting. 
        model.add(Dropout(dropout_value)) # default is 0.01, i.e. 1%
    
         # Add final dense layer with 8 nodes; one for each GOT season 
        model.add(Dense(8, activation = 'softmax')) # we use softmax because it is a classification problem with multiple categories
    
        # Compile model
        model.compile(loss = 'categorical_crossentropy', # categorical because we have more than 2 categories
                      optimizer = optimizer, # optimizer, default = adam
                      metrics = ['accuracy'])
    
        # Print model summary
        model.summary()
    
        # Save model summary
        out_path = os.path.join("..", "output", "cnn_model_summary.txt") # robust path
        with open(out_path, "w") as f:
            with redirect_stdout(f):
                model.summary()
            
        return model 


    def train_model(self, model, X_train_pad, y_train_binarized, n_epochs, X_test_pad, y_test_binarized):
        """
        This method trains the defined CNN model on the training data and validates it on the validation data. 
        """
        # Train model
        history = model.fit(X_train_pad, y_train_binarized,
                            epochs = n_epochs,
                            verbose = True,
                            validation_data = (X_test_pad, y_test_binarized),
                            batch_size = 20)
        
        return model
    
    
    def evaluate_model(self, model, X_test_pad, X_train_pad, y_train_binarized, y_test_binarized, labels, n_epochs):
        """
        This method evaluates the model on the validation data.
        """
        # Predictions
        y_predictions = model.predict(X_test_pad, batch_size=20)
    
        # Classification report
        classification_metrics = classification_report(y_test_binarized.argmax(axis=1),
                                                       y_predictions.argmax(axis=1),
                                                       target_names=sorted(set(labels)))
        
        # Training accuracy
        train_loss, train_accuracy = model.evaluate(X_train_pad, y_train_binarized, verbose=False)
        train_accuracy = round(train_accuracy, 3) # round to 3 decimals
        
        # Validation accuracy
        test_loss, test_accuracy = model.evaluate(X_test_pad, y_test_binarized, verbose=False)
        test_accuracy = round(test_accuracy, 3) # round to 3 decimals
        
        # Save classification report to output directory
        output_path = os.path.join("..", "output", "cnn_classification_metrics.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Below are the classification metrics for the trained model. Number of epochs = {n_epochs}.\nTraining Accuracy: {train_accuracy} \nTesting accuracy: {test_accuracy} \n \n {classification_metrics}")
            
        return classification_metrics
    
    
    def plot_history(self, history, n_epochs):
        """
        This method plots the accuracy and loss curves for the model during training.
        """
        # Produce model history plot
        utils.plot_history(history, epochs = n_epochs) # plot function from utility script
        
        # Save plot to output directory
        plot_path = os.path.join("..", "output", "cnn_model_history.png") # robust path
        plt.savefig(plot_path)


# Define behaviour when called from command line
if __name__=="__main__":
    main()