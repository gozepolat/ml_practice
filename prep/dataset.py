import numpy as np
from prep import morph, util
from six.moves import cPickle
import os


def generate_pretraining_data(lexicon, initial_data, max_words_in_sentence, min_words=25):
    """ generate valid data from the initial data and then generate random word sequences for the invalid data
    :param lexicon: a list of words
    :param initial_data: text entries from real world without labels
    :param max_words_in_sentence: maximum number of words allowed in a random word sentence
    :return: a dataset that contains valid and invalid (random) sentences, should approximately 3x-4x the initial size
    """
    valid_data = generate_valid_data(initial_data, min_words)
    generate_invalid_data(valid_data, lexicon, min_words, max_words_in_sentence)
    np.random.shuffle(valid_data)
    return valid_data


def shorten_sentence(words, min_words):
    """" delete the first few words and if still long, delete the last few words
    :param min_words: minimum number of words that should be preserved
    :param words: a list of words
    :return:
    """
    ix = np.random.randint(1, len(words) - min_words)
    ix2 = len(words)
    if len(words[ix:]) > min_words:
        ix2 = np.random.randint(ix + min_words, len(words))
    return words[ix:ix2]


def generate_valid_data(initial_data, min_words):
    """ generate valid sequences of words by randomly cropping the existing sentences in the initial data

    :param initial_data: has valid sentences
    :param min_words: minimum number of words that should be preserved
    :return: the generated data, mixed with the initial data
    """
    valid_data = []
    for i in initial_data:
        valid_data.append(i + [1])  # add label 1 indicating that it is valid
        if len(i) > min_words + 2:
            valid_data.append(shorten_sentence(i, min_words) + [1])
    return valid_data


def generate_random_sequence(lexicon, min_words, max_words):
    sequence = []
    n = len(lexicon)
    nb_words = np.random.randint(min_words, max_words + 1) + 1
    for i in range(nb_words):
        sequence.append(lexicon[np.random.randint(0, n)])
    return sequence


def generate_invalid_data(valid_data, lexicon, min_words, max_words):
    max_len = len(valid_data)
    for i in range(max_len):
        valid_data.append(generate_random_sequence(lexicon, min_words, max_words) + [0])


def exchange_data(list1, list2, cmp_label, n1=600, n2=600):
    """ fast exchange data
    :param list1:
    :param list2:
    :param cmp_label:
    :param n1:
    :param n2:
    :return:
    """
    if list1 is None or list2 is None or cmp_label is None:
        return list1, list2
    balanced_list1 = []
    balanced_list2 = []
    ctr1 = 0
    ctr2 = 0
    n = 0
    for l1, l2 in zip(list1, list2):
        if len(l1) > 1 and l1[-1] != cmp_label:
            if np.random.uniform(0, 1) < 0.3 and ctr2 < n2:
                balanced_list2.append(l1[:-1] + [cmp_label])
                ctr2 += 1
        if len(l2) > 1 and l2[-1] != cmp_label:
            if np.random.uniform(0, 1) < 0.3 and ctr1 < n1:
                balanced_list1.append(l2[:-1] + [cmp_label])
                ctr1 += 1
        balanced_list1.append(l1)
        balanced_list2.append(l2)
        n += 1
    if n < len(list1):
        for i in range(n, len(list1)):
            l1 = list1[i]
            balanced_list1.append(l1)
            if ctr2 < n2 and len(l1) > 1 and l1[-1] != cmp_label:
                ctr2 += 1
                balanced_list2.append(l1[:-1] + [cmp_label])
    if n < len(list2):
        for i in range(n, len(list2)):
            l2 = list2[i]
            balanced_list1.append(l2)
            if ctr1 < n1 and len(l2) > 1 and l2[-1] != cmp_label:
                ctr1 += 1
                balanced_list1.append(l2[:-1] + [cmp_label])
    np.random.shuffle(balanced_list1)
    np.random.shuffle(balanced_list2)
    return balanced_list1, balanced_list2


def generate_pos_neg_set(neg_X_train, neg_y_train, pos_X_train, pos_y_train):
    """ prepare a binary classification data from pos_train and neg_train sets
    :param neg_train: only the training samples from the negative dataset
    :param pos_train: only the training samples from the positive dataset
    :return: combined dataset for pos/neg classification problem
    """
    pos_neg = []
    ctr1 = 0
    ctr2 = 0
    for i in range(len(neg_y_train)):
        if neg_y_train[i][-1] != 1:  # label is not other
            pos_neg.append(neg_X_train[i] + [0])
            ctr1 += 1
    for i in range(len(pos_y_train)):
        if pos_y_train[i][-1] != 1 and ctr2 < ctr1:  # label is not other
            pos_neg.append(pos_X_train[i] + [1])
            ctr2 += 1
    np.random.shuffle(pos_neg)
    return pos_neg


def dump_all():
    """ preprocess the datasets and pickle dump all the processed data ready for pretraining and training
    pre_y, pre_X, pos_X_train, pos_X_test, pos_y_train, pos_y_test, neg_X_train, neg_X_test, neg_y_train, neg_y_test
    """
    if os.path.isfile("data/pos.pkl") and os.path.isfile("data/pos.pkl"):
        print("directly loading the preprocessed data..")
        word_int_map = cPickle.load(open("data/word_int_map.pkl", "r"))
        pos = cPickle.load(open("data/pos.pkl", "r"))
        neg = cPickle.load(open("data/neg.pkl", "r"))
        pos_test = cPickle.load(open("data/pos_test.pkl", "r"))
        neg_test = cPickle.load(open("data/neg_test.pkl", "r"))
    else:
        # courtesy to http://apps.timwhitlock.info/emoji/tables/unicode
        emoji_dict = cPickle.load(open("data/emoji_dict.pkl", "r"))

        pos_train = "data/pos_emotions_train_emoji_fixed.csv"
        neg_train = "data/neg_emotions_train_emoji_fixed.csv"
        neg_test = "data/neg_emotions_test_emoji_fixed.csv"
        pos_test = "data/pos_emotions_test_emoji_fixed.csv"

        # fix most of the unicode chars and emojis in the data, fix labeling and endline issues
        if not os.path.isfile("data/pos_emotions_train_emoji_fixed.csv"):
            print("fixing pos_emotions_train..")
            pos_train = morph.fix("data/pos_emotions_train.csv", 1, emoji_dict)
        if not os.path.isfile("data/neg_emotions_train_emoji_fixed.csv", ):
            print("fixing neg_emotions_train..")
            neg_train = morph.fix("data/neg_emotions_train.csv", 1, emoji_dict)
        if not os.path.isfile("data/neg_emotions_test_emoji_fixed.csv"):
            print("fixing neg_emotions_test..")
            neg_test = morph.fix("data/neg_emotions_test.csv", 0, emoji_dict)
        if not os.path.isfile("data/pos_emotions_test_emoji_fixed.csv"):
            print("fixing pos_emotions_test..")
            pos_test = morph.fix("data/pos_emotions_test.csv", 0, emoji_dict)

        # tokenize, stem, and generate a corpus from the whole data
        corpus = []
        word_freq = {}
        file_ix = [0]  # indicates w
        ix = 0
        print("tokenization and stemming phase..")
        for path in [pos_test, neg_test, pos_train, neg_train]:  # careful! do not change the order!
            print(path)
            n = util.create_corpus(path, corpus, lower=True)
            print(n)
            ix += n
            file_ix.append(ix)

        # generate a word frequency dictionary from the corpus
        util.freq_corpus(corpus, word_freq)

        # create a mapping between integers and words
        word_int_map = util.word2int(word_freq)  # bidirectional map, unk words <=> 2, 0 & 1 are reserved

        # create an int corpus, by replacing the words with integers
        int_corpus = util.word_corpus2int_corpus(corpus, word_int_map)

        del corpus  # no need for the word version of the corpus anymore
        pos_test = int_corpus[0:file_ix[1]]
        neg_test = int_corpus[file_ix[1]:file_ix[2]]
        # balance the "Other" class and increase the size of the neg/pos training datasets by ~1000 ;)
        pos,neg=exchange_data(int_corpus[file_ix[2]:file_ix[3]], int_corpus[file_ix[3]:],word_int_map["other"])
        # do not balance the "Other" category
        # pos, neg = int_corpus[file_ix[2]:file_ix[3]], int_corpus[file_ix[3]:]
        np.random.shuffle(pos)  # would normally do this in exchange data
        np.random.shuffle(neg)  # would normally do this in exchange data

        cPickle.dump(pos, open("data/pos.pkl", "w"))
        cPickle.dump(neg, open("data/neg.pkl", "w"))
        cPickle.dump(word_int_map, open("data/word_int_map.pkl", "w"))
        cPickle.dump(pos_test, open("data/pos_test.pkl", "w"))
        cPickle.dump(neg_test, open("data/neg_test.pkl", "w"))

    # print(len(pos))
    # print(len(pos_test))
    # print(len(neg))
    # print(len(neg_test))

    max_words_in_sentence = max(max([len(p) for p in pos]), max([len(p) for p in neg]))
    # print(max_words_in_sentence)
    print(word_int_map.left_to_right.items())

    # WARNING! class order of "Other" always must be the last
    # split into X (features = int) and y (class labels = int) and convert y into one hot encoding, for pos data
    pos_X_train, pos_X_test, pos_y_train, pos_y_test = util.reshape_train(pos,
                                                                          [word_int_map["joy"], word_int_map["desir"],
                                                                           word_int_map["love"],
                                                                           word_int_map["other"]], test_size=0.2)
    # WARNING! class order of "Other" always must be the last
    # split into X (features = int) and y (class labels = int) and convert y into one hot encoding, for neg data
    print(word_int_map[159])
    neg_X_train, neg_X_test, neg_y_train, neg_y_test = util.reshape_train(neg,
                                                                          [word_int_map["sad"],
                                                                           word_int_map["anger"],
                                                                           word_int_map["disgust"],
                                                                           word_int_map["hate"],
                                                                           word_int_map["other"]], test_size=0.2)
    if not os.path.isfile("data/pre_data.pkl"):
        # append everything for pretraining
        p_data = pos_X_train + pos_X_test + neg_X_train + neg_X_test
        pre_data = generate_pretraining_data(word_int_map.right_to_left.keys(), p_data,
                                             max_words_in_sentence)
        cPickle.dump(pre_data, open("data/pre_data.pkl", "w"))
    else:
        print("directly loading the pretraining data..")
        pre_data = cPickle.load(open("data/pre_data.pkl", "r"))
    print("generating pos_neg pretraining data..")
    pos_neg = generate_pos_neg_set(neg_X_train, neg_y_train, pos_X_train, pos_y_train)  # no test
    pre_y, pre_X = util.vertical_split(pre_data, -1)
    pos_neg_y, pos_neg_X = util.vertical_split(pos_neg, -1)
    everything = (pre_y, pre_X, pos_neg_y, pos_neg_X, pos_X_train, pos_X_test, pos_y_train, pos_y_test, neg_X_train,
                  neg_X_test, neg_y_train, neg_y_test, max_words_in_sentence)
    print("dumping everything to data/all.pkl..")
    cPickle.dump(everything, open("data/all.pkl", "w"))
    return everything
