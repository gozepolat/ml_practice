import nltk
from prep import morph
# nltk.download("punkt")
# nltk.download("words")
# nltk.download("stopwords")
# nltk.download("twitter_samples")

# fix the datasets
morph.fix_csv("data/pos_emotions_train.csv", 1)
morph.fix_csv("data/neg_emotions_train.csv", 1)
morph.fix_csv("data/neg_emotions_test.csv", 0)
morph.fix_csv("data/pos_emotions_test.csv", 0)

# tokenize and stem
