"""Named Entity Recognition as a classification task.

Author: Kristina Striegnitz and Caleb L'Italien

I affirm that I have carried out my academic endeavors with full
academic honesty. [Caleb L'Italien]

Complete this file for part 1 of the project.
"""
from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

global name_list
VOWELS = ['a', 'e', 'i', 'o', 'u']

def getfeats(word, o):
    """Take a word and its offset with respect to the word we are trying
    to classify. Return a list of tuples of the form (feature_name,
    feature_value).
    """
    o = str(o)
    features = [
        (o + 'word', word),
        (o + 'name gazetteer', __in_name_gazetteer(word)),
        (o + 'has equals sign', __has_equals_sign(word)),
        (o + 'all alphanumeric', __all_alphanumeric(word)),
        (o + 'capitalized', __capitalized(word)),
        (o + 'all caps', __all_caps(word)),
        (o + 'has hyphen', __has_hyphen(word)),
        (o + 'has apostrophe', __has_apostrophe(word)),
        (o + 'number of vowels', __num_vowels(word)),
        (o + 'word length', __word_length(word))
    ]
    return features

def __word_length(word):
    return len(word)

def __num_vowels(word):
    vowels = 0
    for char in word:
        if char in VOWELS:
            vowels += 1
    return vowels

def __has_apostrophe(word):
    for char in word:
        if char == "'":
            return True
    return False

def __has_equals_sign(word):
    for char in word:
        if char == '=':
            return True
    return False

def __all_alphanumeric(word):
    return word.isalnum()

def __capitalized(word):
    return word[0].isupper()

def __all_caps(word):
    for char in word:
        if char.islower():
            return False
    return True

def __has_hyphen(word):
    for char in word:
        if char == '-':
            return True
    return False

def __in_name_gazetteer(word):
    for name in name_list:
        if name.lower() == word.lower():
            return True
    return False

def word2features(sent, i):
    """Generate all features for the word at position i in the
    sentence. The features are based on the word itself as well as
    neighboring words.
    """
    features = []
    # the window around the token (o stands for offset)
    for o in [-1,0,1]:
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            featlist = getfeats(word, o)
            features.extend(featlist)
    return features

if __name__ == "__main__":
    print("\nLoading the data ...")
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))

    print("\nTraining ...")
    train_feats = []
    train_labels = []

    first_words_list = []
    with open('Names_2010Census_Top1000.txt', 'r') as file:
        for line in file:
            first_words_list.append(line.split(" ", 1)[0])
    name_list = first_words_list[33:1035]

    for sent in train_sents:
        for i in range(len(sent)):
            feats = dict(word2features(sent,i))
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    # The vectorizer turns our features into vectors of numbers.
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)
    # Not normalizing or scaling because the example feature is
    # binary, i.e. values are either 0 or 1.

    model = LogisticRegression(max_iter=1400)
    model.fit(X_train, train_labels)

    print("\nTesting ...")
    test_feats = []
    test_labels = []

    # While developing use the dev_sents. In the very end, switch to
    # test_sents and run it one last time to produce the output file
    # results_classifier.txt. That is the results_classifier.txt you
    # should hand in.
    for sent in test_sents:
        for i in range(len(sent)):
            feats = dict(word2features(sent,i))
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = vectorizer.transform(test_feats)
    # If you are normaling and/or scaling your training data, make
    # sure to transform your test data in the same way.
    y_pred = model.predict(X_test)

    print("Writing to results_classifier.txt")
    # format is: word gold pred
    j = 0
    with open("results_classifier.txt", "w") as out:
        for sent in test_sents:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python3 conlleval.py results_classifier.txt")