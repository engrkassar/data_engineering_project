# import libraries

import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
import pickle


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.join(categories.set_index('id'), on='id')	
    
    categories = df['categories'].str.split(';',expand=True)


    category_names = df.iloc[0]['categories'].split(';')
    for i, j in enumerate(category_names):
       category_names[i] = category_names[i].split('-')[0]
    
    with open('category_names.pkl', 'wb') as f:
       pickle.dump(category_names, f)
    print(category_names)
    return df


def clean_data(df):
    with open('category_names.pkl', 'rb') as f:
        category_names = pickle.load(f)
    categories.columns = category_names
    df = messages.join(categories.set_index('id'), on='id')
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.get(-1)

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    df.drop(['categories'], axis=1, inplace=True)
    df = df.join(categories, on='id')
    df = df[df.duplicated()==False]


    return df


def save_data(df, database_filename):
    engine_path = 'sqlite:///' + database_filepath
    engine_write = create_engine(engine_path)
    df.to_sql('data', engine_write, index=False)
    return  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
