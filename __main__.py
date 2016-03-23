import nltk
from prep import morph
import pickle
from prep import freq

# nltk.download("punkt")
# nltk.download("words")
# nltk.download("stopwords")
# nltk.download("twitter_samples")

# fix the data for labeling and endline issues
morph.fix_csv("data/pos_emotions_train.csv", 1)
morph.fix_csv("data/neg_emotions_train.csv", 1)
morph.fix_csv("data/neg_emotions_test.csv", 0)
morph.fix_csv("data/pos_emotions_test.csv", 0)

emoji_dict = pickle.load("data/emoji_dict.pkl")  # courtesy to https://github.com/fionapigott/emoji-counter/blob/master/emoji_dict.py
# tokenize and stem
morph.tokenize_csv("data/pos_emotions_train.csv", emoji_dict)
