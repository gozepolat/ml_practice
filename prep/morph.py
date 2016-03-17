# -*- coding: utf-8 -*-
# morph.py, a. g. polat, 2016
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer
import re


def tokenize(tweet, twt=TweetTokenizer()):
    decoded = tweet.decode("utf8")
    decoded=prep_tweet(decoded)
    tokenized = twt.tokenize(decoded)
    return tokenized


def prep_tweet(tweet):
    """ modified from an existing code, courtesy to: http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
    :param tweet: a single word from the tweet
    :return: tagged/preprocessed version or the tweet itself
    """
    eyes = r"[8:=;]"
    nose = r"['`\-]?"
    tweet = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " url ", tweet)
    tweet = re.sub(r"/", " / ", tweet)
    tweet = re.sub(r"@\w+", " user ", tweet)
    tweet = re.sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " smile ", tweet)
    tweet = re.sub(r"{}{}p+".format(eyes, nose), " lol face ", tweet)
    tweet = re.sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " sad face ", tweet)
    tweet = re.sub(r"{}{}[\/|l*]".format(eyes, nose)," neutral face ", tweet)
    tweet = re.sub(r"<3", " heart ", tweet)
    tweet = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " number ", tweet)
    tweet = re.sub(r"#", " hashtag ", tweet)
    tweet = re.sub(r"([!?.]){2,}", r"\1 repeat ", tweet)
    tweet = re.sub(r"<+-+", " left arrow ", tweet)
    tweet = re.sub(r"-+>+", " right arrow ", tweet)
    return tweet


def nltk_tokenize(text):
    return nltk.wordpunct_tokenize(text)


def stem(tokenized, stemmer=nltk.stem.PorterStemmer()):
    stemmed = [stemmer.stem(w) for w in tokenized]
    print(Word2Vec(stemmed[0]))
    return stemmed


def learn_twitter():
    from nltk.corpus import twitter_samples
    embeddings = Word2Vec(twitter_samples.sents())
    return embeddings


