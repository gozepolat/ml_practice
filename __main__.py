import nltk
import os
from prep import morph
from six.moves import cPickle
from prep import freq

# nltk.download("punkt")
# nltk.download("words")
# nltk.download("stopwords")
# nltk.download("twitter_samples")

# courtesy to http://apps.timwhitlock.info/emoji/tables/unicode
emoji_dict = cPickle.load("data/emoji_dict.pkl")

# fix most of the unicode chars and emojis in the data, as well as labeling and endline issues
if not os.path.isfile("data/pos_emotions_train_emoji_fixed.csv"):
    pos_train = morph.fix("data/pos_emotions_train.csv", 1, emoji_dict)
if not os.path.isfile("data/neg_emotions_train_emoji_fixed.csv", ):
    neg_train = morph.fix("data/neg_emotions_train.csv", 1, emoji_dict)
if not os.path.isfile("data/neg_emotions_test_emoji_fixed.csv"):
    neg_test = morph.fix("data/neg_emotions_test.csv", 0, emoji_dict)
if not os.path.isfile("data/pos_emotions_test_emoji_fixed.csv"):
    pos_test = morph.fix("data/pos_emotions_test.csv", 0, emoji_dict)

# balance the "Other" class and increase the size of the neg/pos training datasets by ~1000 ;)



# tokenize and stem
for f in [pos_train,neg_train,neg_test,pos_test]:
    morph.tokenize_csv(f, emoji_dict)

# get word frequencies, and replace single occurrences with UNK

# construct frequency dictionary



