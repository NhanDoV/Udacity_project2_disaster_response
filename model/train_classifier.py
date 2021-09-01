import sys
import pandas as pd
from sqlalchemy.engine import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import pickle

import nltk
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """
        This function will load the dataset from a database_path in the processing stage,
        Return 
            X: dataframe of messages
            y : dataframe of 36 categories output
            category_names : name of 36 categories
    """

    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df = pd.read_sql_table(database_filepath, engine)
    X = df['message'].values
    y = df.iloc[:, 4:].values
    category_names = list(df.iloc[:, 4:].columns)
    
    return X, y, category_names 

def tokenize(text):
    """
        This function will normalize, removed stop words, stemmed and lemmatized.
        Returns tokenized text
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
        This function will set up a pipeline which prepare for training model
    """

    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()) )
                ])

    parameters = {  'vect__max_df': (0.5, 0.75, 1.0),
                    'clf__estimator__n_estimators': [10, 20, 50],
                    'clf__estimator__min_samples_split': [2, 3, 5]
              }
    cv = GridSearchCV(pipeline, param_grid= parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
        This function will print out the information of accuracy, precision, recall scores of 36 categories
    """
    y_pred = model.predict(X_test)
    
    for k in range(len(category_names)):
        print('Number of category ', k, '\t name: ', category_names[k], '.\n')
        print('\t Accuracy = ', (y_pred[:, k] == Y_test[:,k]).mean(),
              '\t % Precision: \t', precision_score(Y_test[:,k], y_pred[:,k]),
              '\t % Recall : \t', recall_score(Y_test[:,k], y_pred[:,k]),
              '\t % F1_score : \t', f1_score(Y_test[:,k], y_pred[:,k])
             )

def save_model(model, model_filepath):
    """ Save model's best_estimator_ using pickle"""
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33)
        
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