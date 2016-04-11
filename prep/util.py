import numpy as np

import nltk
from sklearn import cross_validation
from sklearn import preprocessing

from prep import morph

np.random.seed(27)  # for reproducibility

'''def exchange_data(path1, path2, n1=1000, n2=1000):
    balanced1 = path1[0:path1.rfind(".")] + "_balanced.csv"
    balanced2 = path2[0:path2.rfind(".")] + "_balanced.csv"
    balanced_list1 = []
    balanced_list2 = []
    ctr1 = 0
    ctr2 = 0
    with open(path1, "r") as t1, open(path2, "r") as t2:
        for line1, line2 in zip(t1, t2):
            l1 = line1.strip()
            l2 = line2.strip()
            if len(l1) > 6 and l1[-6:] != ",Other":
                if np.random.uniform(0, 1) < 0.3 and ctr2 < n2:
                    balanced_list2.append(l1[0:l1.rfind(",")] + ",Other")
                    ctr2 += 1
            if len(l2) > 6 and l2[-6:] != ",Other":
                if np.random.uniform(0, 1) < 0.3 and ctr1 < n1:
                    balanced_list1.append(l2[0:l2.rfind(",")] + ",Other")
                    ctr1 += 1
            balanced_list1.append(l1)
            balanced_list2.append(l2)
    np.random.shuffle(balanced_list1)
    np.random.shuffle(balanced_list2)
    with open(balanced1, "w") as b1:
        for line in balanced_list1:
            b1.write(line + "\n")
    with open(balanced2, "r") as b2:
        for line in balanced_list2:
            b2.write(line + "\n")
    return balanced1, balanced2'''


def vertical_split(list_of_lists, split_index):
    split1 = [values[split_index:]
              for values in list_of_lists]
    split2 = [values[:split_index]
              for values in list_of_lists]
    assert (len(split2) == len(split1))
    return split1, split2


def reshape_train(train, classes, test_size, random_state=27):
    raw_y, X = vertical_split(train, -1)
    y = preprocessing.label_binarize(raw_y, classes=classes)
    X_tr, X_test, y_tr, y_test = cross_validation.train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_tr, X_test, y_tr, y_test


def create_corpus(path, corpus=[], lower=True, ignore_numbers=True, no_header=False):
    nb_lines = 0
    stemmer = nltk.stem.PorterStemmer()
    with open(path, "r") as text:
        skipped = no_header  # skip the first line
        for line in text:
            if skipped:
                stemmed = morph.stem(morph.tokenize(line.strip(), lower=lower, ignore_numbers=ignore_numbers),
                                     stemmer=stemmer)
                # print(" ".join(stemmed))
                corpus.append(stemmed)
                nb_lines += 1
            else:
                skipped = True
    return nb_lines


def freq_corpus(corpus, word_freq={}, doc_freq=None):
    for stemmed in corpus:
        for k in stemmed:  # update the word frequency dictionary
            word_freq[k] = word_freq.setdefault(k, 0) + 1
        if doc_freq is not None:  # can be useful, if tf-idf is required later
            for k in set(stemmed):
                doc_freq[k] = doc_freq.setdefault(k, 0) + 1


class Mapper:
    """Mapper between key_type and other non key_type hashables"""

    def __init__(self, key_type=basestring):
        self.left_to_right = {}
        self.right_to_left = {}
        self.key_type = key_type

    def __getitem__(self, key):
        if key in self.left_to_right:
            return self.left_to_right[key]
        elif key in self.right_to_left:
            return self.right_to_left[key]
        else:
            return None

    def __setitem__(self, key, value):
        """left to right, right to left, keep key_type keys at left side"""
        if isinstance(key, self.key_type):
            self.left_to_right[key] = value
            self.right_to_left[value] = key
        else:
            self.left_to_right[value] = key
            self.right_to_left[key] = value

    def remove(self, k):
        self.right_to_left.pop(self.left_to_right.pop(k))

    def get(self, k, default):
        if k in self.left_to_right:
            return self.left_to_right[k]
        elif k in self.right_to_left:
            return self.right_to_left[k]
        else:
            return default


def word2int(word_freq, simplified_freq=None):
    ix = 3
    word_map = Mapper()
    word_map["<unk>"] = 2  # unk
    for k, v in word_freq.items():
        if v > 4:
            word_map[k] = ix
            ix += 1
            if simplified_freq is not None:
                simplified_freq[k] = v
        elif simplified_freq is not None:
            simplified_freq["<unk>"] = simplified_freq.setdefault("<unk>", 0) + 1
    return word_map


def word_corpus2int_corpus(word_corpus, word_map):
    int_corpus = []
    for words in word_corpus:
        int_corpus.append([word_map.get(word, 2) for word in words])  # 2 for oov
    return int_corpus


"""def reshape_str_output(y_train, mapper, nb_classes):
    mapper = Mapper()
    misc.init_mapper(mapper)
    return reshape_int_output(misc.map_from_mapper(y_train, mapper, left=True), nb_classes)


def map_from_mapper(y_train, mapper, left=True):
    for i, row in enumerate(y_train):
        for j, val in enumerate(row):
            if left:
                if mapper.left_to_right[y_train[i][j]] is None:
                    y_train[i][j] = mapper.left_to_right['other']  # default don't know class
                else:
                    y_train[i][j] = mapper.left_to_right[y_train[i][j]]
            else:
                y_train[i][j] = mapper.right_to_left[y_train[i][j]]
    return y_train"""
