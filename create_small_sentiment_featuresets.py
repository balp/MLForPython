import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer
import os.path

PICKLE_FILE_NAME = 'small_sentiment_set.pickle'

lemmatizer = WordNetLemmatizer()
hm_lines = 100000

def create_lexicon(pos,neg):
    print("create_lexicon({},{}):".format(pos, neg))
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l)
                lexicon += list(all_words)
    print(len(lexicon))
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    print(len(lexicon))
    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print(len(l2))
    return l2

def sample_handling(sample, lexicon, classification):
    feature_set = []

    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            feature_set.append([features, classification])
    return feature_set

def create_feature_set_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling(pos, lexicon, [1, 0])
    features += sample_handling(neg, lexicon, [0, 1])
    random.shuffle(features)
    features = np.array(features)
    testing_size = int(test_size * len(features))
    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])
    return train_x, train_y, test_x, test_y


def get_pos_net_train_and_test_set():
    if os.path.exists(PICKLE_FILE_NAME):
        with open(PICKLE_FILE_NAME, 'rb') as f:
            [train_x, train_y, test_x, test_y] = pickle.load(f)
    else:
        train_x, train_y, test_x, test_y \
            = create_feature_set_and_labels('pos.txt',
                                            'neg.txt')
        with open(PICKLE_FILE_NAME, 'wb') as f:
            pickle.dump([train_x, train_y, test_x, test_y], f)
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, test_x, test_y \
        = get_pos_net_train_and_test_set()
