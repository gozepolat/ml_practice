# -*- coding: utf-8 -*-
# morph.py, a. g. polat, 2016
import nltk
from nltk.tokenize import TweetTokenizer
import wordsegment
import re
import io
import os
#import pandas as pd


def segment_hashtag(h):
    """segment the words inside the hashtag h, discard non alphanum chars"""
    if hasattr(h, "group"):
        h = h.group()[1:]
    else:
        h = h[1:]
    # print(h, " hashtag " + wordsegment.segment(h) + " . ")
    return " hashtag " + " ".join(wordsegment.segment(h)) + " , "


def ends_with_label(line):
    """return whether it ends with a label, ",", or something else
    :param line: it may be encoded in utf-8,
    :return: -1 not a full sample, 0 no label, 1 with a label
    """
    if line is None:
        return -1
    ix = line.rfind(",")
    if ix < 0 or len(line.strip()) < 2:  # discard empty lines
        return -1
    if ix + 1 >= len(line):
        return 0
    if line[ix + 1].isupper() and " " not in line[ix + 1:]:
        if ix + 2 < len(line) and line[ix + 2:].islower():
            return 1
    return 0


def fix_csv(path, k):
    """concatenate the lines without labels to one with a label
    :param path: csv file path to open
    :param k: indicator of three possibilities, -1 not a full sample, 0 no label (test), 1 with a label (train)
    :return: fixed csv file path
    """
    fixed_name = path[0:path.rfind(".")] + "_fixed.csv"
    with open(path, "r") as csv, open(fixed_name, "w") as fixed_csv:
        fixed_line = ""
        in_quote = False
        for line in csv:
            if "\"" in line:
                j = 0
                for m in line:
                    if m == "\"":
                        j += 1
                if j % 2 == 1:
                    in_quote = in_quote is False  # flip the value
            if ends_with_label(line) == k and not in_quote:
                fixed_csv.write(fixed_line + line)
                fixed_line = ""
            else:
                fixed_line += line.strip() + " "
                # print(fixed_line)
    #print(fixed_line)
    return fixed_name


def fix_emoji(path, emoji):
    lines = []
    emoji_path = path[0:path.rfind(".")] + "_emoji.csv"
    if os.path.isfile(emoji_path):
        print("Using the previously created emoji free file:%s" % emoji_path)
        return emoji_path
    try:
        with io.open(path, "r", encoding="utf-8-sig") as csv:
            not_first = False
            for line in csv:
                if not_first:
                    lines.append(line.encode("utf-8-sig"))
                else:  # Skip the first line
                    not_first = True
        with open(emoji_path, "w") as emoji_csv:
            # replace emojis with words
            for i in range(len(lines)):
                for key in emoji.keys():
                    lines[i] = lines[i].replace(key, emoji[key]).decode('unicode_escape').encode('ascii', 'ignore')
            for line in lines:
                emoji_csv.write(line)
    except Exception as e:
        print("Can not preprocess the emojis in the file! There was an exception:")
        print(e)
        import traceback
        print(traceback.format_exc())
        emoji_path = path
    finally:
        return emoji_path


def fix(path, k, emoji):
    emoji_path = fix_emoji(path, emoji)
    print("Replaced emojis with word descriptions. Fixing the labeling issues now..")
    if emoji_path is not None:
        return fix_csv(emoji_path, k)


def prep_tweet(tweet, segment=False):
    """ modified from an existing code from: http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
    :param tweet: a single word from the tweet
    :return: tagged/preprocessed version or the tweet itself
    """
    flags = re.VERBOSE | re.DOTALL | re.LOCALE
    eyes = r"[8:=;\^]"
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
    tweet = re.sub(r"<3|♡ ", " heart ", tweet, flags)
    tweet = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " number ", tweet, flags)
    tweet = re.sub(r"\*([^\*]+)\*", r"\1", tweet, flags)
    tweet = re.sub(r"#", " #", tweet, flags)
    tweet = re.sub(r"(([^a-z]+ [^a-z]+ )+)", r"\1 shout ", tweet, flags)
    tweet = re.sub(r"([!?.]){2,}", r"\1 repeat ", tweet, flags)
    tweet = re.sub(r"([aoe]*h[aoe]+){2,}", " laugh ", tweet, flags | re.IGNORECASE)
    tweet = re.sub(r"<+-+", " from ", tweet, flags)
    tweet = re.sub(r"-+>+", " to ", tweet, flags)
    tweet = re.sub(r"([.,:;!?])", " \1 ", tweet, flags)
    tweet = re.sub(r"zz[z]+", " sleep ", tweet, flags)
    if segment:
        tweet = re.sub(r"#\S+", segment_hashtag, tweet, flags)
    tweet = re.sub(r"([^^])-([^^])]", " ", tweet, flags)
    tweet = re.sub(r"'", " ' ", tweet, flags)
    return tweet


def tokenize(tweet, twt=TweetTokenizer(reduce_len=True), lower=True):
    """convert given string into a list of tokenized words

    :param tweet: tweet string
    :param twt: tweet tokenizer
    :return: list of tokenized words
    """
    # decoded = unicode(tweet)  # .decode("utf-8")
    decoded = prep_tweet(tweet, segment=True)
    if lower:
        tokenized = twt.tokenize(decoded.lower())
    else:
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


'''def construct_unsupervised_dataset(path_list):
    with io.open("../data/unsupervised.csv", "w", encoding="utf-8-sig") as dataset:
        for p in path_list:
            with io.read_csv(p, "r", encoding="utf-8-sig") as csv:
                for line in csv:
                    tokenized = tokenize(line.strip())
                    stemmed = stem(tokenized)
                    dataset.write(unicode(" ").join(stemmed))
                    dataset.write(unicode("\n"))


construct_unsupervised_dataset(["/home/agp/Downloads/pos_emotions_train.csv", "/home/agp/Downloads/neg_emotions_train.csv",
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
