# -*- coding: utf-8 -*-
# morph.py, a. g. polat, 2016
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer
from nltk.corpus import words
import re


def tokenize(tweet, twt=TweetTokenizer()):
    decoded = tweet.decode("utf8")
    decoded = prep_tweet(decoded)
    tokenized = twt.tokenize(decoded)
    return tokenized


MAX_WORD_LENGTH = 100
ENGLISH_WORDS = set(words.words())
TWITTER_WORDS = set()


def add_to_lexicon(tweet_csv, TWITTER_WORDS):
    with open(tweet_csv, "r", encoding="utf8") as f:
        for line in f:
            tokenized = tokenize(line.strip())
            [TWITTER_WORDS.add(t) for t in tokenized]


add_to_lexicon("path", TWITTER_WORDS)


def segment_hashtag(x):
    x = x.group()[1:]
    return "start hashtag " + segment(x, ENGLISH_WORDS) + " end hashtag "


def is_a_word(word, known_words):
    w = word.lower()
    n = len(w)
    if n < 1:
        return False
    elif n == 1:
        if w == 'i' or w == 'u' or w == 'a' or 'm' or 's' or 'd':
            return True
        else:
            return False
    elif w in TWITTER_WORDS:
        return True
    else:
        return w in known_words


def segment(x, known_words):
    """if  there is a known word in x, put a space after it, x has no space"""
    n = len(x)
    for j in range(0, n):
        if j > MAX_WORD_LENGTH:
            return ''
        if is_a_word(x[0:j + 1], known_words):
            if j == n - 1:
                return x  # the final word
            remainder = segment(x[j + 1:], known_words)
            if remainder != '':
                return x[0:j + 1] + ' ' + remainder
    return greedy_segment(x, known_words)


def greedy_segment(x, known_words):
    """ discard the remainder if it is gibberish
    :param x:
    :param known_words:
    :return:
    """
    n = len(x)
    for j in range(0, n):
        if j > MAX_WORD_LENGTH:
            break
        if x[0:j + 1] in known_words:
            if j == n - 1:
                return x  # the whole word
            remainder = segment(x[j + 1:], known_words)
            if remainder != '':
                return x[0:j + 1] + ' ' + remainder
    return x


# print(segment("helloworld", ENGLISH_WORDS))


def prep_tweet(tweet, segment=False):
    """ modified from an existing code, courtesy to: http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
    :param tweet: a single word from the tweet
    :return: tagged/preprocessed version or the tweet itself
    """
    flags = re.IGNORECASE | re.VERBOSE | re.DOTALL | re.LOCALE
    eyes = r"[8:=;]"
    nose = r"['`\-]?"
    tweet = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " url ", tweet, flags)
    tweet = re.sub(r"/", " / ", tweet, flags)
    tweet = re.sub(r"@\w+", " user ", tweet, flags)
    tweet = re.sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " smile ", tweet, flags)
    tweet = re.sub(r"{}{}p+".format(eyes, nose), " laugh ", tweet, flags)
    tweet = re.sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " sad ", tweet, flags)
    tweet = re.sub(r"{}{}[\/|l*]".format(eyes, nose), " neutral ", tweet, flags)
    tweet = re.sub(r" x[xo]+", " kiss ", tweet, flags)
    tweet = re.sub(r"<3", " heart ", tweet, flags)
    tweet = re.sub(r"♡", " heart ", tweet, flags)
    tweet = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " number ", tweet, flags)
    tweet = re.sub(r"\*([^\*]+)\*", r"\1", tweet, flags)
    tweet = re.sub(r"#", " #", tweet, flags)
    if segment:
        tweet = re.sub(r"#\S+", segment_hashtag, tweet, flags)
    tweet = re.sub(r"([!?.]){2,}", r"\1 repeat ", tweet, flags)
    tweet = re.sub(r"([aoe]*h[aoe]+){2,}", " laugh ", tweet, flags)
    tweet = re.sub(r"<+-+", " from ", tweet, flags)
    tweet = re.sub(r"-+>+", " to ", tweet, flags)
    return tweet


def nltk_tokenize(text):
    return nltk.wordpunct_tokenize(text)


def stem(tokenized, stemmer=nltk.stem.PorterStemmer()):
    stemmed = [stemmer.stem(w) for w in tokenized]
    # print(Word2Vec(stemmed[0]))
    return stemmed


def learn_twitter():
    from nltk.corpus import twitter_samples
    embeddings = Word2Vec(twitter_samples.sents())
    return embeddings
