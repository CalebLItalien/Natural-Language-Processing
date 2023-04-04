"""Text classification for identifying complex words.

Author: Kristina Striegnitz and Caleb L'Italien

I affirm that I have carried out my academic endeavors with full
academic honesty. [Caleb L'Italien]

Complete this file for parts 2-4 of the project.

"""

from collections import defaultdict

import gzip
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from syllables import count_syllables
from nltk.corpus import wordnet as wn

from evaluation import get_fscore, evaluate

def load_file(data_file):
    """Load in the words and labels from the given file."""
    words = []
    labels = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels


### 2.1: A very simple baseline

def all_complex(data_file):
    """Label every word as complex. Evaluate performance on given data set. Print out
    evaluation results."""
    words, labels = load_file(data_file)
    predictions = [1] * len(words)
    evaluate(predictions, labels)


### 2.2: Word length thresholding3

def word_length_threshold(training_file, development_file):
    """Find the best length threshold by f-score and use this threshold to classify
    the training and development data. Print out evaluation results."""
    files = [training_file, development_file]
    length_threshold = 8
    for file in files:
        words, labels = load_file(file)
        predictions = []
        for word in words:
            if len(word) >= length_threshold:
                predictions.append(1)
            else:
                predictions.append(0)
        evaluate(predictions, labels)


### 2.3: Word frequency thresholding

def load_ngram_counts(ngram_counts_file):
    """Load Google NGram counts (i.e. frequency counts for words in a
    very large corpus). Return as a dictionary where the words are the
    keys and the counts are values.
    """
    counts = defaultdict(int)
    with gzip.open(ngram_counts_file, 'rt') as f:
        for line in f:
            token, count = line.strip().split('\t')
            if token[0].islower():
                counts[token] = int(count)
    return counts

def word_frequency_threshold(training_file, development_file, counts):
    """Find the best frequency threshold by f-score and use this
    threshold to classify the training and development data. Print out
    evaluation results.
    """
    frequency_threshold = 1883
    files = [training_file, development_file]
    for file in files:
        words, labels = load_file(file)
        predictions = []
        for word in words:
            if counts[word] >= frequency_threshold:
                predictions.append(1)
            else:
                predictions.append(0)
        evaluate(predictions, labels)


### 3.1: Naive Bayes

def naive_bayes(training_file, development_file, counts):
    """Train a Naive Bayes classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    words, train_labels = load_file(training_file)
    train_features = np.array([[len(word), counts[word]] for word in words])

    words_dev, dev_labels = load_file(development_file)
    dev_features = np.array([[len(word), counts[word]] for word in words_dev])

    train_features_scaled = (train_features - train_features.mean(axis = 0)) / train_features.std(axis = 0)
    dev_features_scaled = (dev_features - train_features.mean(axis = 0)) / train_features.std(axis = 0)

    clf = GaussianNB()
    clf.fit(train_features_scaled, train_labels)
    train_pred = clf.predict(train_features_scaled)
    dev_pred = clf.predict(dev_features_scaled)

    evaluate(train_pred, train_labels)
    evaluate(dev_pred, dev_labels)


### 3.2: Logistic Regression

def logistic_regression(training_file, development_file, counts):
    """Train a Logistic Regression classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    words, train_labels = load_file(training_file)
    train_features = np.array([[len(word), counts[word]] for word in words])

    words_dev, dev_labels = load_file(development_file)
    dev_features = np.array([[len(word), counts[word]] for word in words_dev])

    train_features_scaled = (train_features - train_features.mean(axis = 0)) / train_features.std(axis = 0)
    dev_features_scaled = (dev_features - train_features.mean(axis = 0)) / train_features.std(axis = 0)

    clf = LogisticRegression()
    clf.fit(train_features_scaled, train_labels)
    train_pred = clf.predict(train_features_scaled)
    dev_pred = clf.predict(dev_features_scaled)
    
    evaluate(train_pred, train_labels)
    evaluate(dev_pred, dev_labels)


### 3.3: Build your own classifier

def my_classifier(training_file, development_file, counts):
    """Train a Decision Tree or Random Forest classifier using number of synonyms,
    number of synonyms, frequency, length, number of common letters, and number of uncommon
    letters features. Print out evaluation results on the development data.
    """
    words, train_labels = load_file(training_file)
    train_features = np.array([[len(wn.synsets(word)), count_syllables(word), counts[word], len(word), __common_letters(word), __uncommon_letters(word)] for word in words])

    words_dev, dev_labels = load_file(development_file)
    dev_features = np.array([[len(wn.synsets(word)), count_syllables(word), counts[word], len(word), __common_letters(word), __uncommon_letters(word)] for word in words_dev])

    train_features_scaled = (train_features - train_features.mean(axis = 0)) / train_features.std(axis = 0)
    dev_features_scaled = (dev_features - train_features.mean(axis = 0)) / train_features.std(axis = 0)

    clf = tree.DecisionTreeClassifier()
    clf.fit(train_features_scaled, train_labels)
    dev_pred = clf.predict(dev_features_scaled)

    evaluate(dev_pred, dev_labels)

def __test_on_unlabeled(training_file, development_file, test_file, counts):
    words, train_labels = load_file(training_file)
    train_features = np.array([[len(wn.synsets(word)), count_syllables(word), counts[word], len(word), __common_letters(word), __uncommon_letters(word)] for word in words])

    words_dev, dev_labels = load_file(development_file)
    dev_features = np.array([[len(wn.synsets(word)), count_syllables(word), counts[word], len(word), __common_letters(word), __uncommon_letters(word)] for word in words_dev])

    train_features_scaled = (train_features - train_features.mean(axis = 0)) / train_features.std(axis = 0)
    dev_features_scaled = (dev_features - train_features.mean(axis = 0)) / train_features.std(axis = 0)

    test_words, labels = load_file(test_file)
    test_features = np.array([[len(wn.synsets(word)), count_syllables(word), counts[word], len(word), __common_letters(word), __uncommon_letters(word)] for word in test_words])

    clf = RandomForestClassifier()
    clf.fit(train_features_scaled, train_labels)
    clf.fit(dev_features_scaled, dev_labels)

    test_predictions = clf.predict(test_features)
    with open('text_labels.txt', 'w') as out_file:
        for prediction in test_predictions:
            out_file.write("%s\n" % str(prediction))
    out_file.close()

def __common_letters(word):
    common_letters = ['e', 'a', 'r', 'i', 'o', 't', 'n', 's', 'l', 'c']
    common_letters_count = 0
    for char in word:
        if char in common_letters:
            common_letters_count += 1
    return common_letters_count

def __uncommon_letters(word):
    uncommon_letters = ['j', 'q', 'x', 'z', '-']
    uncommon_letters_count = 0
    for char in word:
        if char in uncommon_letters:
            uncommon_letters_count += 1
    return uncommon_letters_count

def baselines(training_file, development_file, counts):
    print("========== Baselines ===========\n")

    print("Majority class baseline")
    print("-----------------------")
    print("Performance on training data")
    all_complex(training_file)
    print("\nPerformance on development data")
    all_complex(development_file)

    print("\nWord length baseline")
    print("--------------------")
    word_length_threshold(training_file, development_file)

    print("\nWord frequency baseline")
    print("-------------------------")
    print("max ngram counts:", max(counts.values()))
    print("min ngram counts:", min(counts.values()))
    word_frequency_threshold(training_file, development_file, counts)

def classifiers(training_file, development_file, counts):
    print("\n========== Classifiers ===========\n")

    print("Naive Bayes")
    print("-----------")
    naive_bayes(training_file, development_file, counts)

    print("\nLogistic Regression")
    print("-----------")
    logistic_regression(training_file, development_file, counts)

    print("\nMy classifier")
    print("-----------")
    my_classifier(training_file, development_file, counts)


if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    print("Loading ngram counts ...")
    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)

    baselines(training_file, development_file, counts)
    classifiers(training_file, development_file, counts)

    ## YOUR CODE HERE
    # Train your best classifier, predict labels for the test dataset and write
    # the predicted labels to the text file 'test_labels.txt', with ONE LABEL
    # PER LINE
    __test_on_unlabeled(training_file, development_file, test_file, counts)