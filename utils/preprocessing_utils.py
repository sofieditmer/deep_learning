#!/usr/bin/env python
"""
Stores preprocessing functions. 
"""

# Dependencies
import os
import sys
sys.path.append(os.path.join(".."))
import pandas as pd
import nltk
nltk.download('punkt')
from collections import OrderedDict

# Function for loading and tokenizing the data
def load_and_tokenize(filename):
    """
    This function loads the data and tokenizes it using the NLTK module. 
    It also removes short sentences, since these do not make much sense to keep. 
    """
    # Load data
    data_df = pd.read_csv(filename, lineterminator = "\n")
    
    # Keep only relevant columns
    data_df = data_df.loc[:, ("Season", "Sentence")]
        
    # Since some lines are long and some are short, it makes sense to tokenize them into sentences. 
    # For this I use the sent_tokenize function from NLTK
    data_df["Tokenized_Sentences"] = data_df["Sentence"].apply(nltk.sent_tokenize)
    
    # Now the lines have been sentence tokenized. 
    # Now I want to make sure that there is one sentece per row which will make it easier to chunk them later. 
    # For this I use the explode function which puts each sentence into its own row
    # Inspiration for how to solve this was found in this StackOverflow thread:
    # https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-entry-to-separate-rows 
    data_df = data_df.explode("Tokenized_Sentences")
    
    # Since some of the sentences are very short, I want to remove the shortest ones that do not really make sense to keep.
    # I remove sentences that are less than 2 characters
    short_lines = data_df[data_df["Tokenized_Sentences"].str.len()<2].index
    preprocessed_data = data_df.drop(short_lines, axis = 0) # axis = 0 means rows 
    
    return preprocessed_data
    
    
# Function for creating chunks of data 
def chunk_data(preprocessed_data, chunk_size):
    """
    This function takes the sentences and chunks them into chunks. The size of the chunks depends on the chunk size specified by the user.
    """
    # For each season, combine sentences in chunks
    # Inspired by this StackOverflow thread: 
    # https://stackoverflow.com/questions/58713593/concatenate-every-n-rows-into-one-row-pandas 
    for season in preprocessed_data["Season"].unique():
        chunked_preprocessed_df = preprocessed_data.groupby(preprocessed_data.index // chunk_size).agg(' '.join)
    
    # Change name of 'Tokenized Sentences' column, since we now have chunks of sentences within it
    chunked_preprocessed_df.columns = ['Season', 'Sentence', 'Chunks']
    
    # Take only relevant columns
    chunked_preprocessed_df = chunked_preprocessed_df.loc[:, ("Season", "Chunks")]
    
    # Remove duplicates in 'Season' column
    chunked_preprocessed_df["Season"] = chunked_preprocessed_df["Season"].str.split().apply(lambda x: ','.join(OrderedDict.fromkeys(x).keys()))
    
    # Replace comma with space
    chunked_preprocessed_df["Season"] = chunked_preprocessed_df["Season"].str.replace(',',' ')
    
    # Because I have chunked the data, the seasons will overlap when changing from one season to another.
    # Hence, I remove these cases to make sure that there are no overlapping seasons
    chunked_preprocessed_df = chunked_preprocessed_df[~chunked_preprocessed_df['Season'].isin(['Season 1 2'])]
    chunked_preprocessed_df = chunked_preprocessed_df[~chunked_preprocessed_df['Season'].isin(['Season 2 3'])]
    chunked_preprocessed_df = chunked_preprocessed_df[~chunked_preprocessed_df['Season'].isin(['Season 3 4'])]
    chunked_preprocessed_df = chunked_preprocessed_df[~chunked_preprocessed_df['Season'].isin(['Season 4 5'])]
    chunked_preprocessed_df = chunked_preprocessed_df[~chunked_preprocessed_df['Season'].isin(['Season 5 6'])]
    chunked_preprocessed_df = chunked_preprocessed_df[~chunked_preprocessed_df['Season'].isin(['Season 6 7'])]
    chunked_preprocessed_df = chunked_preprocessed_df[~chunked_preprocessed_df['Season'].isin(['Season 7 8'])]

    return chunked_preprocessed_df
            
    
if __name__=="__main__":
    pass