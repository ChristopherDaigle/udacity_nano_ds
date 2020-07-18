__author__ = "Chris"
import nltk
import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
nltk.download(['punkt', 'wordnet'])


def load_data():
    """
    Function to import relevant data from corporate_messaging.csv
    :return X (numpy.ndarray): factors
    :return y (numpy.ndarray): labels
    """
    df = pd.read_csv('corporate_messaging.csv', encoding='latin-1')
    df = df[(df["category:confidence"] == 1) & (df['category'] != 'Exclude')]
    X = df.text.values
    y = df.category.values

    return X, y


def tokenize(text: str) -> list:
    """
    Function to clean text

    - Replaces URLs with "urlplaceholder"
    - Tokenizes
    - Lemmatizes
    - Removes extra whitespace
    - Transforms to lowercase

    :param text: string data
    :return clean_tokens: list of cleaned tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
    clean_tokens = [lemmatizer.lemmatize(token, pos='v') for token in clean_tokens]

    return clean_tokens


def transformer(X, y):
    """
    Function to transform data

    - Train-Test Split
    - Vectorize
    - TF-IDF

    :param X: factor data
    :param y: label data
    :return X_train_tfidf: tfidf of training data
    :return y_train: training labels
    :return X_test_tfidf: tfidf of testing data
    :return y_test: testing labels
    """
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    vect = CountVectorizer(tokenizer=tokenize)
    tfidf = TfidfTransformer()

    X_train_count = vect.fit_transform(X_train)
    X_train_tfidf = tfidf.fit_transform(X_train_count)

    X_test_count = vect.transform(X_test)
    X_test_tfidf = tfidf.transform(X_test_count)

    return X_train_tfidf, y_train, X_test_tfidf, y_test


def trainer(X_train_tfidf, y_train, X_test_tfidf, clf):
    """
    Train model on data

    - Fit the classifier
    - Predict with testing labels

    :param X_train_tfidf: tfidf of training data
    :param y_train: training labels
    :param X_test_tfidf: tfidf of testing data
    :param clf: classifier algorithm
    :return y_pred: predictions from model
    """
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    return y_pred


def display_results(y_test, y_pred):
    """
    Display results of trained classifier

    - Create array of unique labels
    - Create confusion matrix
    - Calculate accuracy
    - Display confusion matrix as dataframe
    - Print accuracy score
    :param y_test: testing labels
    :param y_pred: predictions from model
    :return None: displays model information
    """
    labels = np.array(list(set(y_test)), dtype='object')
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = accuracy_score(y_test, y_pred)

    print(pd.DataFrame(confusion_mat,
                         columns=[lab + "_true" for lab in labels],
                         index=[lab + "_pred" for lab in labels]))
    print("Accuracy:", round(accuracy, 4))

    return None


def main(clf):
    """
    Complete all components of the ML Pipeline

    :param clf: Classifier to be fitted
    :return None: displays results
    """
    # 1
    X, y = load_data()
    # 2, 3
    X_train_tfidf, y_train, X_test_tfidf, y_test = transformer(X, y)
    y_pred = trainer(X_train_tfidf=X_train_tfidf,
                     y_train=y_train,
                     X_test_tfidf=X_test_tfidf,
                     clf=clf)
    # 4
    display_results(y_test, y_pred)

    return None


if __name__ == "__main__":
    main(clf=RandomForestClassifier(random_state=0))