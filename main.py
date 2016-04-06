import os
import numpy as np
from sklearn import preprocessing
from dl import models
from dl import benchmark
from sklearn import svm  # future work: fusion
import cPickle

from prep import dataset

# nltk.download("punkt")
# nltk.download("words")
# nltk.download("stopwords")
# nltk.download("twitter_samples")


if __name__ == "__main__":
    if not os.path.isfile("data/all.pkl"):
        (pre_y, pre_X, pos_neg_y, pos_neg_X, pos_X_train, pos_X_test, pos_y_train, pos_y_test, neg_X_train, neg_X_test,
        neg_y_train, neg_y_test, max_words_in_sentence) = dataset.dump_all()
        pass
    else:
        print("loading preprocessed data..")
        (pre_y, pre_X, pos_neg_y, pos_neg_X, pos_X_train, pos_X_test, pos_y_train, pos_y_test, neg_X_train, neg_X_test,
         neg_y_train, neg_y_test, max_words_in_sentence) = cPickle.load(open("data/all.pkl", "r"))
    print("data is ready for training!")


    def train(initial_model, X_train, X_test, y_train, y_test, nb_epoch=16,
              batch_size=30, max_words=max_words_in_sentence, evaluate=True):
        return models.train_model(initial_model, X_train, y_train, X_test, y_test, max_words=max_words,
                                  nb_epoch=nb_epoch, batch_size=batch_size, evaluate=evaluate)


    # word_int_map=cPickle.load(open("data/word_int_map.pkl"))
    # print(len(word_int_map.left_to_right))
    # pretraining
    pre_model = models.construct_pre_model(max_words=max_words_in_sentence)
    print("constructed initial model")
    if not os.path.isfile("data/pos_neg_model.hdf5"):
        if not os.path.isfile("data/pre_model.hdf5"):  # train valid/invalid model
            split_ix = int(len(pre_X) * (1 - 0.2))
            score, acc = train(pre_model, pre_X[:split_ix], pre_X[split_ix:], pre_y[:split_ix], pre_y[split_ix:],
                               nb_epoch=1)
            pre_model.save_weights("data/pre_model.hdf5", overwrite=True)
            print("(pre)training of valid/invalid set is completed with validation loss and accuracy", (score, acc))
        else:
            # print("loading the pretrained embedding layer weights")
            pre_model.load_weights("data/pre_model.hdf5")
        split_ix = int(len(pos_neg_X) * (1 - 0.2))
        score, acc = train(pre_model, pos_neg_X[:split_ix], pos_neg_X[split_ix:], pos_neg_y[:split_ix],
                           pos_neg_y[split_ix:], nb_epoch=2)
        pre_model.save_weights("data/pos_neg_model.hdf5", overwrite=True)
        print("(pre)training of pos/neg set is completed with validation loss and accuracy", (score, acc))
        # print("(this model can be directly used for positive vs negative sentiment classification)")
    else:
        print("loading the pretrained embedding layer weights")
        pre_model.load_weights("data/pos_neg_model.hdf5")

    print("finished loading the pretrained embedding layer weights")

    # scores without pretrained weights
    # stratified cross-validation 10-fold as default, give n_fold=k to benchmark.cross_validate to change it
    evaluate = False  # do not calculate validation score in keras since it will be done extensively later
    lb = preprocessing.LabelBinarizer()
    # neg model
    print("starting cross-validation for neg model")
    neg_labels = ["Sadness", "Anger", "Disgust", "Hate", "Other"]
    lb.fit(neg_labels)
    neg_y_labels = benchmark.remap(np.concatenate((neg_y_train, neg_y_test)), neg_labels)
    neg_X = np.concatenate((neg_X_train, neg_X_test))
    avg_scores = np.zeros((4, 5))
    cm_history = []

    benchmark.cross_validate(len(neg_labels), neg_X, neg_y_labels, 10, "neg", neg_labels, avg_scores, cm_history,
                             max_words=max_words_in_sentence, pretrained=pre_model.layers[0])
    print("end of cross-validation for negative task")
    avg_cm = cm_history[0]
    for cm in cm_history[1:]:
        avg_cm += cm
    avg_cm /= 10
    print("total confusion matrix")
    print(avg_cm)
    print("average confusion matrix")
    print(avg_cm)
    print("average precision, recall, fscore and support values for each class:")
    print(", ".join(neg_labels))
    print(avg_scores / 10.0)
    # pos model
    print("starting cross-validation for pos model")
    pos_labels = ["Joy", "Desire", "Love", "Other"]
    lb = None
    lb = preprocessing.LabelBinarizer(sparse_output=True)
    lb.fit(pos_labels)
    # pos_y = lb.inverse_transform(np.concatenate((pos_y_train, pos_y_test)))
    pos_y_labels = benchmark.remap(np.concatenate((pos_y_train, pos_y_test)), pos_labels)
    pos_X = np.concatenate((pos_X_train, pos_X_test))
    avg_scores = np.zeros((4, 4))
    cm_history = []
    benchmark.cross_validate(len(pos_labels), pos_X, pos_y_labels, 10, "pos", pos_labels, avg_scores, cm_history,
                             max_words=max_words_in_sentence, pretrained=pre_model.layers[0])
    print("end of cross-validation for positive task")
    avg_cm = cm_history[0]
    for cm in cm_history[1:]:
        avg_cm += cm
    print("total confusion matrix")
    print(avg_cm)
    avg_cm /= 10
    print("average confusion matrix")
    print(avg_cm)
    print("average precision, recall, fscore and support values for each class:")
    print(", ".join(pos_labels))
    print(avg_scores / 10.0)

    """ # scores with pretrained weights, check this scores later, pretraining should make it better
    print("scores with pretrained weights:")
    # train the dataset on a cnn / lstm architecture
    pos_model = models.construct_cnn_lstm(nb_class=4, stateful=False, convolutional=True,
                                          max_words=max_words_in_sentence,
                                          pretrained_embedding=pre_model.layers[0])
    score, acc = train(pos_model, pos_X_train, pos_X_test, pos_y_train, pos_y_test, nb_epoch=3)

    # use the same embedding layer for neg?
    print("positive model is trained with validation loss and accuracy", (score, acc))

    neg_model = models.construct_cnn_lstm(nb_class=5, stateful=False, convolutional=True,
                                          max_words=max_words_in_sentence,
                                          pretrained_embedding=pre_model.layers[0])
    score, acc = models.train_model(neg_model, neg_X_train, neg_X_test, neg_y_train, neg_y_test, nb_epoch=3)
    print("negative model is trained with validation loss and accuracy", (score, acc))
    """
