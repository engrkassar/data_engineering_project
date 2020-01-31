# import libraries

import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import argparse

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
import pickle
from sqlalchemy import create_engine



def model_input():

    parser = argparse.ArgumentParser()

    parser.add_argument('database_filepath', action="store", default='../data/DisasterResponse.db', type=str)
    parser.add_argument('model_filepath', action="store", default='classifier.pkl', type=str)
    
    results = parser.parse_args()

    return results


def load_data(database_filepath):
    engine_path = 'sqlite:///' + database_filepath
    engine_read = create_engine(engine_path)
    df = pd.read_sql("SELECT * FROM data", engine_read)
    df.dropna(inplace=True)
    X = df.iloc[:,2].values
    y = df.iloc[:,4:].values
    with open('../data/category_names.pkl', 'rb') as f:
        category_names = pickle.load(f)
    return X, y, category_names


def tokenize(text):

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
   
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ])),

        ('clf', MultiOutputClassifier(MultinomialNB()))
    ])

    parameters = {
       
        'features__text_pipeline__vect__ngram_range': [(1, 1), (1, 2)],
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__n_jobs': [1, 5],

    }


    return pipeline



def evaluate_model(model, X_test, y_test, category_names):

    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print("Accuracy:", accuracy)
    print(classification_report(y_test[:,1:], y_pred[:,1:], target_names=labels))
    return


def save_model(model, model_filepath):
    model_pkl_filename = model_filepath
    # Open the file to save as pkl file
    model_pkl = open(model_pkl_filename, 'wb')
    pickle.dump(model, model_pkl)
    # Close the pickle instances
    model_pkl.close()
    return

def main():
    model_input()
    model_input().database_filepath
    model_input().model_filepath
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
