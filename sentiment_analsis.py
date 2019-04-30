import csv
import math
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sys

# Static variables
training_percentage = 0.75
training_dataset_file_path = './movie_data/train.txt'
testing_dataset_file_path = './movie_data/test.txt'

'''
Function reads file into Python given file path

Parameters
  dataset_file_path | string | The file path for file to read into Python.
                                Delimiter is newline

Returns
  list[string]
'''
def import_dataset_by_path(dataset_file_path):
    data = []
    with open(dataset_file_path, 'r') as dataset_file:
        for document in dataset_file:
            document = document.rstrip('\n')
            data.append(document)
    return data

'''
Function cleans up text of raw text included in it

Parameters
  review_text | string | the text of document review

Returns
  string | processed text with cleaned up raw tags
'''
def preprocess_text(review_text):
    COMPACT_REGEX = re.compile("[.;:!\'?,\"()\[\]]")
    REMOVE_TAG_REGEX = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    # ALways lower case the text
    text_lower = review_text.lower()
    # Remove punctuations
    rem_punctuations_text = COMPACT_REGEX.sub("", text_lower)
    # Remove tags
    rem_tags_text = REMOVE_TAG_REGEX.sub(" ", rem_punctuations_text)
    return rem_tags_text

'''
Function cleans English stop words from the review text

Parameters
  review_text | string | the review phrase document

Returns
  string
'''
eng_stop_words = stopwords.words('english')

def clean_stop_words(review_text):
    lowercase_text = review_text.lower()
    non_stopwords_lst = [w for w in lowercase_text.split() if w not in eng_stop_words]
    non_stopwords_str = ' '.join(non_stopwords_lst)
    return non_stopwords_str

'''
Function

Parameters
  review_text | string | the review phrase document

Returns
  string
'''
def stem_text(review_text):
    stemmer = PorterStemmer()
    stemmed_text_list = [stemmer.stem(word) for word in review_text.split()]
    stemmed_text_str = ' '.join(stemmed_text_list)
    return stemmed_text_str

'''
Function

Parameters
  review_text | string | the review phrase document

Returns
  string
'''
def lemmatize_text(review_text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text_list = [lemmatizer.lemmatize(word) for word in review_text.split()]
    lemmatized_text_str = ' '.join(lemmatized_text_list)
    return lemmatized_text_str


'''
Function splits dataset into training and testing partition
If the number of rows is not integer, round up

Returns
  tuple | (training_partition, testing_partition, size of partition)
'''
def partition_dataset_with_target(dataset, target_val, train_percent):
    if (train_percent < 0 or train_percent > 1):
        print("Expecting a train partition percentage value of 0-1. Received {}".format(train_percent))
        sys.exit(1)
    return train_test_split(dataset, target_val, train_size=train_percent)

def tfidf_vectorize(fit_dataset, dataset=None):
    tfidf_vector = TfidfVectorizer()
    tfidf_vector.fit(fit_dataset)
    return tfidf_vector.transform(dataset or fit_dataset)

def run_logistic_regression_model(X_train, X_target, y_train, y_target):
    print("Running Logistic Regression...")
    lr = None;
    lr = LogisticRegression(C=1)
    lr.fit(X_train, X_target)
    print ("Accuracy for C=1: %s" % (accuracy_score(y_target, lr.predict(y_train))))
    return lr

'''
Function handles running the entire program for sentiment analysis on the moview review Dataset
1) Import data set
2) Tokenize and remove stopwords
3) Perform stemming/lemmatizing
4) Generate the Tf-idf matrix
5) Partition data to training and testing partition
6) Perform training on model with training partition
7) Test accuracy on validation partition
8) Print accuracy
'''
def run_sentiment_analysis_new():
    ## Import training dataset
    print("Importing training dataset...")
    training_dataset = import_dataset_by_path(training_dataset_file_path)

    # Preprocess text
    print("Preprocessing training dataset...")
    preprocessed_dataset = [preprocess_text(review_text) for review_text in training_dataset]

    # Clean stopwords from each movie review
    print("Cleaning training dataset...")
    cleaned_stop_words_dataset = [clean_stop_words(review_text) for review_text in preprocessed_dataset]

    print("Normalizing training dataset...")
    # Stem Text
    # stemmed_text_dataset = [stem_text(review_text) for review_text in cleaned_stop_words_dataset]

    # Lemmatize Text
    lemmatized_text_dataset = [lemmatize_text(review_text) for review_text in cleaned_stop_words_dataset]

    # Vectorize dataset
    X = tfidf_vectorize(lemmatized_text_dataset)
    # Break training partition into training and testing
    # 1: positive sentiment
    # 0: negative sentiment
    target_variable = [1 if i < 12500 else 0 for i in range(0, 25000)]
    X_train, X_test, y_train, y_test = partition_dataset_with_target(
        X,
        target_variable,
        train_percent=training_percentage
    )

    # Train model with training dataset
    # print("Running sentiment analysis on partitioned training dataset...")
    # run_logistic_regression_model(X_train, y_train, X_test, y_test)

    ## Import testing dataset
    print("Importing testing dataset...")
    testing_dataset = import_dataset_by_path(testing_dataset_file_path)

    # Preprocess text
    print("Preprocessing testing dataset...")
    preprocessed_testing_dataset = [preprocess_text(review_text) for review_text in testing_dataset]

    # Clean stopwords from each movie review
    print("Cleaning testing dataset...")
    cleaned_stop_words_testing_dataset = [clean_stop_words(review_text) for review_text in preprocessed_testing_dataset]

    print("Normalizing testing dataset...")
    # Stem Text
    # stemmed_text_testing_dataset = [stem_text(review_text) for review_text in cleaned_stop_words_testing_dataset]

    # Lemmatize Text
    lemmatized_text_testing_dataset = [lemmatize_text(review_text) for review_text in cleaned_stop_words_testing_dataset]

    # Vectorize dataset
    X_test = tfidf_vectorize(lemmatized_text_dataset, lemmatized_text_testing_dataset)
    print("Running sentiment analysis on training and testing dataset...")
    run_logistic_regression_model(X, target_variable, X_test, target_variable)

if __name__ == "__main__":
    run_sentiment_analysis_new()
