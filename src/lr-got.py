#!/usr/bin/env python
"""
Info: This script creates a baseline logistic regression model and trains it on the dialogue of all Game of Thrones 8 seasons to predict which season a given line is from. This model can be used as a means of evaluating how a deep learning model performs. 

Parameters:
    (optional) input_file: str <name-of-input-file>, default = "game_of_thrones_script.csv"
    (optional) chunk_size: int <size-of-chunks>, default = 10
    (optional) test_size: float <size-of-test-data>, default = 0.25
    (optional) main_output_filename: str <name-of-output-file>, default = "lr_classification_report.txt"

Usage:
    $ python lr-got.py
    
Output:
    - lr_classification_report.txt: classification report of the logistic regression classifier.
    - lr_heatmap.png: normalized heatmap displaying an overview of the performance of the logistic regression classifier.
    - lr_cross_validation_results.png: cross-validation results obtained by the logistic regression classifier.
"""

### DEPENDENCIES ###

# core libraries
import os
import sys
sys.path.append(os.path.join(".."))

# matplotlib
import matplotlib.pyplot as plt

# import utility functions
import utils.classifier_utils as clf # for classification
import utils.preprocessing_utils as preprocess # for preprocessing

# Machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit # cross-validation
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

# argparse
import argparse

import warnings
warnings.filterwarnings('ignore')


### MAIN FUNCTION ###

def main():
    
    ### ARGPARSE ###
    
    # Initialize ArgumentParser class
    ap = argparse.ArgumentParser()
    
    # Argument 1: Path to training data
    ap.add_argument("-i", "--input_filename",
                    type = str,
                    required = False, # this argument is not required
                    help = "Path to the training data",
                    default = "game_of_thrones_script.csv")

    # Argument 2: Test data size 
    ap.add_argument("-ts", "--test_size",
                    type = float,
                    required = False, # this argument is not required
                    help = "Define the size of the test dataset as a float value, e.g. 0.20",
                    default = 0.25) # default test size
    
    # Argument 3: Size of chunks
    ap.add_argument("-c", "--chunk_size",
                    type = int,
                    required = False, # this argument is not required
                    help = "Number of lines to chunk together",
                    default = 10) # default chunk size

    # Argument 4: Name of output file
    ap.add_argument("-o", "--main_output_filename",
                    type = str,
                    required = False, # this argument is not required
                    help = "Define the name of the main output file, i.e. the classification report",
                    default = "lr_classification_report.txt") # default name
    
    # Parse arguments
    args = vars(ap.parse_args())
    
    # Save input parameters
    input_file = os.path.join("..", "data", args["input_filename"])
    test_size = args["test_size"]
    chunk_size = args["chunk_size"]
    output_filename = args["main_output_filename"]

    # Create output directory
    if not os.path.exists(os.path.join("..", "output")):
        os.mkdir(os.path.join("..", "output"))

    # User message
    print("\n[INFO] Initializing the construction of the logistic regression classifier...")
    
    # Load data
    print(f"\n[INFO] Loading, preprocessing, and chunking '{input_file}'...")
    preprocessed_data = preprocess.load_and_tokenize(input_file)
    preprocessed_df = preprocess.chunk_data(preprocessed_data, chunk_size)
    
    # Instantiate the logistic regression classifier class
    lr = Logistic_regression(preprocessed_df)
    
    # Create test-train split
    print(f"\n[INFO] Creating train-test split with test size of {test_size}...")
    X_train, X_test, y_train, y_test, sentences, labels = lr.create_train_test(test_size)
    
    # Vectorize data
    print("\n[INFO] Vectorizing training and validation data using a count vectorizer...")
    vectorizer, X_train_feats, X_test_feats = lr.vectorize(X_train, X_test)
    
    # Perform grid search
    print("\n[INFO] Performing grid search to estimate the most optimal hyperparameters...")
    best_params = lr.perform_gridsearch(X_train_feats, y_train)
    
    # Train logistic regression classifier on training data
    print("\n[INFO] Building logistic regression classifier and training it on the traning data...")
    classifier = lr.train_lr_classifier(X_train_feats, y_train, best_params)
    
    # Evaluate the logistic regression classifier
    print("\n[INFO] Evaluating the logistic regression classifier on the validation data...")
    classification_metrics = lr.evaluate_lr_classifier(classifier, X_test_feats, y_test, output_filename, best_params)
    print(f"\n[INFO] Below are the classification metrics for the logistic regression classifier trained with the following hyperparameters: {best_params}. These metrics are also saved as '{output_filename} in 'output' directory. \n \n {classification_metrics}\n")
    
    # Cross-validation
    print("\n[INFO] Performing cross-validation...")
    lr.cross_validate(sentences, vectorizer, labels, test_size)
    
    # User message
    print("\n[INFO] Done! You have now defined and trained a logistic regression classifier. The results can be found in the 'output' directory.\n")
    
    
### LOGISTIC REGRESSION ###
    
# Creating Logistic Regression class 
class Logistic_regression:
    
    # Intialize Logistic regression class
    def __init__(self, preprocessed_df):
        
        # Receive input
        self.preprocessed_df = preprocessed_df
        
        
    def create_train_test(self, test_size):
        """
        This method creates X_train, X_test, y_train, and y_test based on the preprocesssed and chunked dialogue.
        """
        # Extract seasons as labels
        labels = self.preprocessed_df['Season'].values
        
        # Create train data
        sentences = self.preprocessed_df["Chunks"].values

        # Create training and test split using sklearn
        X_train, X_test, y_train, y_test = train_test_split(sentences,
                                                            labels,
                                                            test_size=test_size,
                                                            random_state=42)
    
        return X_train, X_test, y_train, y_test, sentences, labels


    def vectorize(self, X_train, X_test):
        """
        This method vectorizes the training and test data using a CountVectorizer available in scikit-learn.
        """
        # Intialize count vectorizer with default parameters
        vectorizer = CountVectorizer()
        
        # Fit vectorizer to training and test data
        X_train_feats = vectorizer.fit_transform(X_train)
        X_test_feats = vectorizer.transform(X_test)
        
        # Normalize features
        X_train_feats = preprocessing.normalize(X_train_feats, axis=0)
        X_test_feats = preprocessing.normalize(X_test_feats, axis=0)

        return vectorizer, X_train_feats, X_test_feats
    
    
    def perform_gridsearch(self, X_train_feats, y_train):
        """
        This method performs grid search, i.e. iterates over possible hyperparameters for the logistic regression model
        in order to find the most optimal values. 
        The hyperparameters that I have chosen to iterate over are C (regularization strength/learning rate) and tolerance. 
        The smaller the regularization value, the stronger the regularization, and the longer the training time.
        The tolerance value tells the optimization algorithm when to stop, which means that if the tolerance value is high
        the algorithm stops before it converges. Hence, the tolerance value should not be too high, because this means that 
        the model might not converge.
        """
        # Initialize pipeline consisting of the "classifier" which is made up of the logistic regression classification function
        pipe = Pipeline([('classifier', LogisticRegression())])

        # Set tunable parameters for grid search
        C = [1.0, 0.1, 0.01]             # regularization strengths
        tol = [0.1, 0.01, 0.001]         # tolerance values

        # Create parameter grid (a Python dictionary) that contains the hyperparameters 
        parameters = dict(classifier__C = C,
                          classifier__tol = tol)

        # Choose which metrics on which we want to optimize
        scores = ['precision', 'recall', 'f1']
        
        # For each of the metrics find the optimal hyperparameter values
        for score in scores:
            
            # Initialise Gridsearch with predefined parameters
            clf = GridSearchCV(pipe, 
                               parameters, 
                               scoring= f"{score}_weighted",
                               cv=5) # using 10-fold cross-validation
            
            # Fit grid search model to data
            clf.fit(X_train_feats, y_train)

            # Print the best paremeters to terminal 
            print(f"\n [INFO] Best parameters found on training data for '{score}' metric: \n {clf.best_params_} \n")

            # Save best parameters
            best_params = clf.best_params_
            
        return best_params
            
     
    def train_lr_classifier(self, X_train_feats, y_train, best_params):
        """
        This method trains the logistic regression classifier on the training data with hyperparameters estimated by grid search.
        """ 
        # Train the logistic regression classifier on the scaled data
        classifier = LogisticRegression(random_state = 42,
                                        max_iter = 10000,
                                        C=best_params['classifier__C'], # taking the most optimal regularization strength as estimated by grid search
                                        tol=best_params['classifier__tol'], # taking the best tolerance value estimated by grid search
                                        multi_class='multinomial').fit(X_train_feats, y_train) # when using 'multinomial' the loss minimized is the multinomial loss fit across the entire probability distribution
        
        return classifier
    
    
    def evaluate_lr_classifier(self, classifier, X_test_feats, y_test, output_filename, best_params):
        """
        This method evaluates the logistic regression classifier on the validation data. 
        """
        # Extract predictions
        y_pred = classifier.predict(X_test_feats)
        
        # Evaluate model
        classification_metrics = metrics.classification_report(y_test, y_pred)
          
        # Save in output directory
        out_path = os.path.join("..", "output", output_filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"Below are the classification metrics for the logistic regression classifier trained with the following hyperparameters: {best_params} \n \n{classification_metrics}")
          
        # Plot results as a heatmap using utility function
        clf.plot_cm(y_test, y_pred, normalized=True)
          
        # Save heatmap to output directory
        out_path_heatmap = os.path.join("..", "output", "lr_heatmap.png")
        plt.savefig(out_path_heatmap)
        
        return classification_metrics
    
    
    def cross_validate(self, sentences, vectorizer, labels, test_size):
        """
        This method performs cross-validation and saves results in output directory.
        """
        # Vectorize the sentences
        X_vect = vectorizer.fit_transform(sentences)
       
        # Intialize cross-validation
        title = "Learning Curves (Logistic Regression)"
        cv = ShuffleSplit(n_splits=100, test_size=test_size, random_state=0)
        
        # Run cross-validation
        model = LogisticRegression(random_state=42, max_iter = 10000)
        
        # Plot learning curves
        clf.plot_learning_curve(model, title, X_vect, labels, cv=cv, n_jobs=4)
          
        # save in output directory
        out_path = os.path.join("..", "output", "lr_cross_validation_results.png")
        plt.savefig(out_path)
        
        
# Define behaviour when called from command line
if __name__=="__main__":
    main()