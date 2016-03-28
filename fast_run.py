from six.moves import cPickle
from dl import models
print("directly loading the preprocessed data..")
word_int_map = cPickle.load(open("data/word_int_map.pkl", "r"))
pos = cPickle.load(open("data/pos.pkl", "r"))
neg = cPickle.load(open("data/neg.pkl", "r"))
pos_test = cPickle.load(open("data/pos_test.pkl", "r"))
neg_test = cPickle.load(open("data/neg_test.pkl", "r"))
pre_data = cPickle.load(open("data/pre_data.pkl", "r"))


def vertical_split(list_of_lists, split_index):
    split1 = [values[split_index:]
              for values in list_of_lists]
    split2 = [values[:split_index]
              for values in list_of_lists]
    assert (len(split2) == len(split1))
    return split1, split2

max_words_in_sentence = max(max([len(p) for p in pos]), max([len(p) for p in neg]))
pre_y, pre_X = vertical_split(pre_data, -1)
pre_model = models.construct_pre_model(max_words_in_sentence=max_words_in_sentence)
split_ix = int(len(pre_X) * (1 - 0.2))
score, acc = models.train_model(pre_model, pre_X[:split_ix], pre_X[split_ix:], pre_y[:split_ix], pre_y[split_ix:],
                                nb_epoch=16, batch_size=30, max_words_in_sentence=max_words_in_sentence)
print()


