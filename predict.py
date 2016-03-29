from dl import models
import os
# from prep import dataset
import cPickle
import io
from prep import morph
import numpy as np

if __name__ == '__main__':
    if not os.path.isfile("data/all.pkl"):
        (pre_y, pre_X, pos_neg_y, pos_neg_X, pos_X_train, pos_X_test, pos_y_train, pos_y_test, neg_X_train, neg_X_test,
        neg_y_train, neg_y_test, max_words_in_sentence) = dataset.dump_all()
        pass
    else:
        print("loading preprocessed data..")
        (pre_y, pre_X, pos_neg_y, pos_neg_X, pos_X_train, pos_X_test, pos_y_train, pos_y_test, neg_X_train, neg_X_test,
         neg_y_train, neg_y_test, max_words_in_sentence) = cPickle.load(open("data/all.pkl", "r"))
    print("data is ready for training!")

    pre_model = models.construct_pre_model(max_words=max_words_in_sentence)
    print("constructed initial model")
    if not os.path.isfile("data/pos_neg_model.hdf5"):
        if not os.path.isfile("data/pre_model.hdf5"):  # train valid/invalid model
            split_ix = int(len(pre_X) * (1 - 0.2))
            score, acc = models.train_model(pre_model, pre_X[:split_ix], pre_X[split_ix:], pre_y[:split_ix],
                                            pre_y[split_ix:],
                                            nb_epoch=1, max_words=max_words_in_sentence)
            pre_model.save_weights("data/pre_model.hdf5", overwrite=True)
            print("(pre)training of valid/invalid set is completed with validation loss and accuracy", (score, acc))
        else:
            # print("loading the pretrained embedding layer weights")
            pre_model.load_weights("data/pre_model.hdf5")
        split_ix = int(len(pos_neg_X) * (1 - 0.2))
        score, acc = models.train_model(pre_model, pos_neg_X[:split_ix], pos_neg_X[split_ix:], pos_neg_y[:split_ix],
                                        pos_neg_y[split_ix:], nb_epoch=2, max_words=max_words_in_sentence)
        pre_model.save_weights("data/pos_neg_model.hdf5", overwrite=True)
        print("(pre)training of pos/neg set is completed with validation loss and accuracy", (score, acc))
        # print("(this model can be directly used for positive vs negative sentiment classification)")
    else:
        print("loading the pretrained embedding layer weights")
        pre_model.load_weights("data/pos_neg_model.hdf5")
    # train the dataset on a cnn / lstm architecture
    pos_model = models.construct_cnn_lstm(nb_class=4, stateful=False, convolutional=True,
                                          max_words=max_words_in_sentence,
                                          pretrained_embedding=pre_model.layers[0])
    score, acc = models.train_model(pos_model, pos_X_train,  pos_y_train, pos_X_test, pos_y_test, nb_epoch=40,
                                    max_words=max_words_in_sentence)
    # predict:
    pos_test = cPickle.load(open("data/pos_test.pkl"))
    pos_test = models.pad(pos_test, max_words=max_words_in_sentence)
    pos_pred = pos_model.predict_classes(pos_test)
    pos_labels = [u"Joy", u"Desire", u"Love", u"Other"]
    pred_labels = [pos_labels[i] for i in pos_pred]
    morph.fix_csv("data/pos_emotions_test.csv", 0)
    pos_model.save_weights("data/pos_model.hdf5",overwrite=True)
    with io.open("data/pos_emotions_test_fixed.csv", "r", encoding="utf-8-sig") as test, io.open(
            "data/pos_emotions_pred.csv", "w", encoding="utf-8-sig") as pred:
        i = 0
        for line in test:
            pred.write(line.strip() + pred_labels[i] + u"\n")
            i += 1
    neg_model = models.construct_cnn_lstm(nb_class=5, stateful=False, convolutional=True,
                                          max_words=max_words_in_sentence,
                                          pretrained_embedding=pre_model.layers[0])
    score, acc = models.train_model(neg_model, neg_X_train, neg_y_train, neg_X_test, neg_y_test, nb_epoch=40,
                                    max_words=max_words_in_sentence)
    # predict:
    neg_test = cPickle.load(open("data/neg_test.pkl"))
    neg_test = models.pad(neg_test, max_words=max_words_in_sentence)
    neg_pred = neg_model.predict_classes(neg_test)
    neg_labels = [u"Sadness", u"Anger", u"Disgust", u"Hate", u"Other"]
    pred_labels = [neg_labels[i] for i in neg_pred]
    morph.fix_csv("data/neg_emotions_test.csv", 0)
    neg_model.save_weights("data/neg_model.hdf5", overwrite=True)
    with io.open("data/neg_emotions_test_fixed.csv", "r", encoding="utf-8-sig") as test, io.open(
            "data/neg_emotions_pred.csv", "w", encoding="utf-8-sig") as pred:
        i = 0
        for line in test:
            pred.write(line.strip() + pred_labels[i] + u"\n")
            i += 1
