import nltk
import os
from prep import morph, util
from six.moves import cPickle
from prep import freq
from sklearn.cross_validation import StratifiedKFold

# nltk.download("punkt")
# nltk.download("words")
# nltk.download("stopwords")
# nltk.download("twitter_samples")

# courtesy to http://apps.timwhitlock.info/emoji/tables/unicode
emoji_dict = cPickle.load("data/emoji_dict.pkl")

# fix most of the unicode chars and emojis in the data, fix labeling and endline issues
if not os.path.isfile("data/pos_emotions_train_emoji_fixed.csv"):
    pos_train = morph.fix("data/pos_emotions_train.csv", 1, emoji_dict)
if not os.path.isfile("data/neg_emotions_train_emoji_fixed.csv", ):
    neg_train = morph.fix("data/neg_emotions_train.csv", 1, emoji_dict)
if not os.path.isfile("data/neg_emotions_test_emoji_fixed.csv"):
    neg_test = morph.fix("data/neg_emotions_test.csv", 0, emoji_dict)
if not os.path.isfile("data/pos_emotions_test_emoji_fixed.csv"):
    pos_test = morph.fix("data/pos_emotions_test.csv", 0, emoji_dict)

# balance the "Other" class and increase the size of the neg/pos training datasets by ~1000 ;)
util.exchange_data()

# tokenize, stem, and generate a frequency dictionary from the whole dataset
# and then create a mapping between integers and words

for f in [pos_train,neg_train,neg_test,pos_test]:
    morph.stem(morph.tokenize_csv(f, emoji_dict))
#  replace and save the words as integers

for k,v in freq:
    if v ==1:
        freq["<unk>"]+=1
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

if __name__ == "__main__":
    n_folds = 10
    data, labels, header_info = load_data()
    skf = StratifiedKFold(labels, n_folds=n_folds, shuffle=True)

    for i, (train, test) in enumerate(skf):
            print ("Cross-validation fold: %d/%d"%(i+1, n_folds))
            model = None # Clearing the NN.
            model = construct_cnn_lstm()
            train_and_evaluate_model(model, data[train], labels[train], data[test], labels[test))

