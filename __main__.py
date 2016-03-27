import os
from dl import models
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
# from sklearn import svm  # future work: fusion

import cPickle

# nltk.download("punkt")
# nltk.download("words")
# nltk.download("stopwords")
# nltk.download("twitter_samples")
""" # if all.pkl does not exist
from prep import dataset
if not os.path.isfile("data/all.pkl"):
    dataset.dump_all()
# """

if __name__ == "__main__":
    print("loading preprocessed data..")
    (pre_y, pre_X, pos_neg_y, pos_neg_X, pos_X_train, pos_X_test, pos_y_train, pos_y_test, neg_X_train, neg_X_test,
     neg_y_train, neg_y_test, max_words_in_sentence) = cPickle.load(open("data/all.pkl", "r"))

    print("finished loading!")


    def train(initial_model, X_train, X_test, y_train, y_test, nb_epoch=16,
              batch_size=30, max_words=max_words_in_sentence, evaluate=True):
        return models.train_model(initial_model, X_train, y_train, X_test, y_test, max_words_in_sentence=max_words,
                                  nb_epoch=nb_epoch, batch_size=batch_size, evaluate=evaluate)


    # pretraining
    pre_model = models.construct_pre_model(max_words_in_sentence=max_words_in_sentence)

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
        pre_model.load_weights("data/pos_neg_model.hdf5")  # TODO uncomment
    n_folds = 10  # cross-validation
    lb = preprocessing.LabelBinarizer(sparse_output=True)
    # pos model
    print("starting cross-validation for pos model")
    pos_labels = ["Joy", "Desire", "Love", "Other"]
    lb.fit(pos_labels)
    pos_y = lb.inverse_transform(np.concatenate((pos_y_train, pos_y_test)))
    pos_X = np.concatenate((pos_X_train, pos_X_test))
    skf = StratifiedKFold(pos_y, n_folds=n_folds, shuffle=False, random_state=None)  # already shuffled
    for i, (train_ix, test_ix) in enumerate(skf):
        print ("Cross-validation fold: %d/%d" % (i + 1, n_folds))
        pos_model = None  # Clearing the NN.
        pos_model = models.construct_cnn_lstm(nb_class=4, stateful=False, convolutional=True,
                                              max_words_in_sentence=max_words_in_sentence, )
        # pretrained_embedding=pre_model.layers[0])
        X_train, X_test = pos_X[train_ix], pos_X[test_ix]
        y_train, y_test = pos_y[train_ix], pos_y[test_ix]
        models.train_model(pos_model, X_train, preprocessing.label_binarize(y_train, classes=pos_labels), nb_epoch=1,
                           evaluate=False)
        y_test_pred = pos_model.predict(X_test)
        cm = confusion_matrix(y_test, preprocessing.label_binarize(y_test_pred, classes=pos_labels),
                              labels=pos_labels)  # ["Joy","Desire","Love","Other"]
        print(cm)
        scores = precision_recall_fscore_support(y_test, y_test_pred, average=None, labels=pos_labels)
        print(scores)
        cPickle.dump(cm, open("data/scores/cm_pos_cross_%d.pkl" % i, "w"))
        cPickle.dump(cm, open("data/scores/precision_recall_fscore_support_pos_cross_%d.pkl" % i, "w"))
    # neg model
    print("starting cross-validation for neg model")

    neg_labels = ["Sadness", "Anger", "Disgust", "Hate", "Other"]
    lb.fit(neg_labels)
    neg_y = lb.inverse_transform(np.concatenate((neg_y_train, neg_y_test)))
    neg_X = np.concatenate((neg_X_train, neg_X_test))
    skf = StratifiedKFold(neg_y, n_folds=n_folds, shuffle=False, random_state=None)  # already shuffled
    # precision, recall, f1-score and support on the test data for each class should be recorded and provided, as well as the confusion matrix.
    for i, (train_ix, test_ix) in enumerate(skf):
        print ("Cross-validation fold: %d/%d" % (i + 1, n_folds))
        neg_model = None  # Clearing the NN.
        neg_model = models.construct_cnn_lstm(nb_class=5, stateful=False, convolutional=True,
                                              max_words_in_sentence=max_words_in_sentence, )
        # pretrained_embedding=pre_model.layers[0])
        X_train, X_test = neg_X[train_ix], neg_X[test_ix]
        y_train, y_test = neg_y[train_ix], neg_y[test_ix]
        models.train_model(neg_model, X_train, preprocessing.label_binarize(y_train, classes=neg_labels), nb_epoch=1,
                           evaluate=False)
        y_test_pred = neg_model.predict(X_test)
        cm = confusion_matrix(y_test, preprocessing.label_binarize(y_test_pred, classes=neg_labels),
                              labels=neg_labels)
        print(cm)
        scores = precision_recall_fscore_support(y_test, y_test_pred, average=None, labels=neg_labels)
        print(scores)
        cPickle.dump(cm, open("data/scores/cm_neg_cross_%d.pkl" % i, "w"))
        cPickle.dump(cm, open("data/scores/precision_recall_fscore_support_neg_cross_%d.pkl" % i, "w"))
        # pos model

        """
        >>> y_true = [2, 0, 2, 2, 0, 1]
        >>> y_pred = [0, 0, 2, 2, 0, 2]
        >>> confusion_matrix(y_true, y_pred)
        array([[2, 0, 0],
               [0, 0, 1],
               [1, 0, 2]])

        >>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
        >>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
        >>> confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
        array([[2, 0, 0],
               [0, 0, 1],
               [1, 0, 2]])

        """
        """
        >>> from sklearn.metrics import precision_recall_fscore_support
        >>> y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
        >>> y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
        >>> precision_recall_fscore_support(y_true, y_pred, average='macro')
        ... # doctest: +ELLIPSIS
        (0.22..., 0.33..., 0.26..., None)
        >>> precision_recall_fscore_support(y_true, y_pred, average='micro')
        ... # doctest: +ELLIPSIS
        (0.33..., 0.33..., 0.33..., None)
        >>> precision_recall_fscore_support(y_true, y_pred, average='weighted')
        ... # doctest: +ELLIPSIS
        (0.22..., 0.33..., 0.26..., None)

        It is possible to compute per-label precisions, recalls, F1-scores and
        supports instead of averaging:
        >>> precision_recall_fscore_support(y_true, y_pred, average=None,
        ... labels=['pig', 'dog', 'cat'])
        ... # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
        (array([ 0. ,  0. ,  0.66...]),
         array([ 0.,  0.,  1.]),
         array([ 0. ,  0. ,  0.8]),
         array([2, 2, 2]))"""
        """
    # train the dataset on a cnn / lstm architecture
    pos_model = models.construct_cnn_lstm(nb_class=4, stateful=False, convolutional=True,
                                          max_words_in_sentence=max_words_in_sentence,
                                          pretrained_embedding=pre_model.layers[0])
    score, acc = train(pos_model, pos_X_train, pos_X_test, pos_y_train, pos_y_test, nb_epoch=3)

    # use the same embedding layer for neg?
    print("positive model is trained with validation loss and accuracy", (score, acc))

    neg_model = models.construct_cnn_lstm(nb_class=5, stateful=False, convolutional=True,
                                          max_words_in_sentence=max_words_in_sentence,
                                          pretrained_embedding=pre_model.layers[0])
    score, acc = models.train_model(neg_model, neg_X_train, neg_X_test, neg_y_train, neg_y_test, nb_epoch=3)

    # use the same embedding layer for neg?
    print("negative model is trained with validation loss and accuracy", (score, acc))"""
