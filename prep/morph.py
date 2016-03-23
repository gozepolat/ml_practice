# -*- coding: utf-8 -*-
# morph.py, a. g. polat, 2016
import nltk
from nltk.tokenize import TweetTokenizer
import wordsegment
import re
import io
import pandas as pd



def segment_hashtag(x):
    if hasattr(x, "group"):
        x = x.group()[1:]
    else:
        x = x[1:]
    # print(x, " hashtag " + wordsegment.segment(x) + " . ")
    return " hashtag " + " ".join(wordsegment.segment(x)) + " , "


def ends_with_label(line):
    """return whether it ends with a label, ",", or something else
    :param line: it may be encoded in utf-8,
    :return: -1 not a full sample, 0 no label, 1 with a label
    """
    ix = line.rfind(u",")
    if ix < 0:
        return -1
    if ix + 1 >= len(line):
        return 0
    if line[ix + 1].isupper() and u" " not in line[ix + 1:]:
        if ix + 2 < len(line) and line[ix + 2:].islower():
            return 1
    return 0


def fix_csv(path, k):
    """concatenate the lines without labels to one with a label"""
    with io.open(path, "r", encoding="utf-8-sig") as csv, io.open(path[0:path.rfind(".")] + "_fixed.csv", "w",
                                                                  encoding="utf-8-sig") as fixed_csv:
        fixed_line = u""
        in_quote = False
        for line in csv:
            if u"\"" in line or u"\'" in line:
                j = 0
                for k in line:
                    if k == u"\"" or k == u"\'":
                        j += 1
                if j % 2 == 1:
                    in_quote = in_quote is False  # flip the value
            if ends_with_label(line) == k and not in_quote:
                fixed_csv.write(fixed_line + line)
                fixed_line = u""
            else:
                fixed_line += line.strip() + u" "


def prep_tweet(tweet, segment=False):
    """ modified from an existing code from: http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
    :param tweet: a single word from the tweet
    :return: tagged/preprocessed version or the tweet itself
    """
    flags = re.VERBOSE | re.DOTALL | re.LOCALE
    eyes = r"[8:=;]"
    nose = r"['`\-]?"
    tweet = re.sub(r",", " , ", tweet, flags)
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
    tweet = re.sub(r"([A-Z0-9\)\(,.!?:;]+ )+", r"\1 shout ", tweet, flags)
    tweet = re.sub(r"([!?.]){2,}", r"\1 repeat ", tweet, flags)
    tweet = re.sub(r"([aoe]*h[aoe]+){2,}", " laugh ", tweet, flags | re.IGNORECASE)
    tweet = re.sub(r"<+-+", " from ", tweet, flags)
    tweet = re.sub(r"-+>+", " to ", tweet, flags)
    tweet = re.sub(r"[.]", " . ", tweet, flags)
    if segment:
        tweet = re.sub(r"#\S+", segment_hashtag, tweet, flags)

    return tweet


def tokenize(tweet, twt=TweetTokenizer(reduce_len=True)):
    """convert given string into a list of tokenized words

    :param tweet: tweet string
    :param twt: tweet tokenizer
    :return: list of tokenized words
    """
    decoded = unicode(tweet)  # .decode("utf-8")
    decoded = prep_tweet(decoded, segment=True)
    tokenized = twt.tokenize(decoded)
    return tokenized


def stem(tokenized, stemmer=nltk.stem.PorterStemmer()):
    """ convert the words in the tokenized list into their stemmed forms

    :param tokenized:
    :param stemmer: stemmer model to use
    :return:
    """
    stemmed = [stemmer.stem(w) for w in tokenized]
    return stemmed


def construct_unsupervised_dataset(path_list):
    with io.read("../data/unsupervised.csv", "w", encoding="utf-8-sig") as dataset:
        for p in path_list:
            with io.read_csv(p, "r", encoding="utf-8-sig") as csv:
                for line in csv:
                    tokenized = tokenize(line.strip())
                    stemmed = stem(tokenized)
                    dataset.write(unicode(" ").join(stemmed))
                    dataset.write(unicode("\n"))


'''construct_unsupervised_dataset(["/home/agp/Downloads/pos_emotions_train.csv", "/home/agp/Downloads/neg_emotions_train.csv",
     "/home/agp/Downloads/pos_emotions_test.csv", "/home/agp/Downloads/neg_emotions_test.csv"])

with io.open("../data/unsupervised.csv", "r", encoding="utf-8-sig") as dataset:
    for i in dataset:
        print(repr(i.strip()))
        raw_input("dodo")
# MAX_WORD_LENGTH = 100
# ENGLISH_WORDS = set(words.words())

'''

# print(segment("helloworld", ENGLISH_WORDS))





"""
def nltk_tokenize(text):
    return nltk.wordpunct_tokenize(text)

def learn_twitter():
    from nltk.corpus import twitter_samples
    embeddings = Word2Vec(twitter_samples.sents())
    return embeddings
"""

'''def add_to_lexicon(tweet_csv, TWITTER_WORDS):
    with io.open(tweet_csv, "r", encoding="utf-8-sig") as f:
        for line in f:
            tokenized = tokenize(line.strip())
            [TWITTER_WORDS.add(t) for t in tokenized]

add_to_lexicon("path", TWITTER_WORDS)

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
'''
