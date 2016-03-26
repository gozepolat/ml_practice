import nltk
import os
from prep import morph, util
from six.moves import cPickle
from dl import models

"""from sklearn import cross_validation

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, Imputer
"""

# nltk.download("punkt")
# nltk.download("words")
# nltk.download("stopwords")
# nltk.download("twitter_samples")
if os.path.isfile("data/pos.pkl") and os.path.isfile("data/pos.pkl"):
    print("directly loading the preprocessed data")
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
    pos, neg = util.exchange_data(int_corpus[file_ix[2]:file_ix[3]], int_corpus[file_ix[3]:], word_int_map["other"])
    cPickle.dump(pos, open("data/pos.pkl", "w"))
    cPickle.dump(neg, open("data/neg.pkl", "w"))
    cPickle.dump(word_int_map, open("data/word_int_map.pkl", "w"))
    cPickle.dump(pos_test, open("data/pos_test.pkl", "w"))
    cPickle.dump(neg_test, open("data/neg_test.pkl", "w"))

print(len(pos))
max_words_in_sentence = max([len(p) for p in pos])
print(max_words_in_sentence)
print(word_int_map.left_to_right.items())
# split into X (features = int) and y (class labels = int) and convert y into one hot encoding,
pos_X_train, pos_X_test, pos_y_train, pos_y_test = util.reshape_train(pos, [word_int_map["joy"], word_int_map["desir"],
                                                                            word_int_map["love"],
                                                                            word_int_map["other"]], test_size=0.2)
# train the dataset on a cnn / lstm architecture
pos_model = models.construct_cnn_lstm(nb_class=4, stateful=True, max_words_in_sentence=max_words_in_sentence)
score, acc = models.train_model(pos_model, pos_X_train, pos_X_test, pos_y_train, pos_y_test, nb_epoch=6,
                                batch_size=30, max_words_in_sentence=max_words_in_sentence)

# use the same embedding layer for neg?
print(score, acc)
"""
# divide the data into labels (y_*) and features (X_*)
X_train=None
y_train=None
with open(pos_train,"r") as f:
    train=[line.strip() for line in f]


#

# get word frequencies, and replace single occurrences with UNK

# construct frequency dictionary



def load_data():
    # load your data using this function

def create model():
    # create your model using this function

def train_and_evaluate__model(model, data[train], labels[train], data[test], labels[test)):
    model.fit...
    # fit and evaluate here.

# 10-fold stratified cross-validation
if __name__ == "__main__":
    n_folds = 10
    data, labels, header_info = load_data()
    skf = StratifiedKFold(labels, n_folds=n_folds, shuffle=True)

    for i, (train, test) in enumerate(skf):
            print ("Cross-validation fold: %d/%d"%(i+1, n_folds))
            model = None # Clearing the NN.
            model = construct_cnn_lstm()
            train_and_evaluate_model(model, data[train], labels[train], data[test], labels[test))

"""

"""
pos_X_train = []
pos_y_train = []
pos_X_test = []
pos_y_test = []
neg_X_train = []
neg_y_train = []
neg_X_test = []
neg_y_test = []

X_train = []
y_train = []
X_test = []
y_test = []
"""
