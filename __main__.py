import nltk
from prep import morph
import pickle
from prep import freq

# nltk.download("punkt")
# nltk.download("words")
# nltk.download("stopwords")
# nltk.download("twitter_samples")

# courtesy to http://apps.timwhitlock.info/emoji/tables/unicode
emoji_dict = pickle.load("data/emoji_dict.pkl")  # courtesy to https://github.com/fionapigott/emoji-counter/blob/master/emoji_dict.py

# fix the unicode chars and emojis in the data, labeling and endline issues
morph.fix("data/pos_emotions_train.csv", 1, emoji_dict)
morph.fix("data/neg_emotions_train.csv", 1, emoji_dict)
morph.fix("data/neg_emotions_test.csv", 0, emoji_dict)
morph.fix("data/pos_emotions_test.csv", 0, emoji_dict)


# tokenize and stem
morph.tokenize_csv("data/pos_emotions_train.csv", emoji_dict)
