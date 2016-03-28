from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from dl import models
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
import cPickle
import numpy as np
import copy


def cross_validate(nb_class, X, y, nb_epoch, task, labels, avg_scores, n_folds=10, evaluate=False, max_words=100,
                   stateful=False, convolutional=True, pretrained=None):
    skf = StratifiedKFold(y, n_folds=n_folds, shuffle=False, random_state=None)  # already shuffled
    for i, (train_ix, test_ix) in enumerate(skf):
        print ("Cross-validation fold: %d/%d" % (i + 1, n_folds))
        model = None  # Clearing the NN.
        model = models.construct_cnn_lstm(stateful=stateful, convolutional=convolutional, nb_class=nb_class,
                                          max_words=max_words, pretrained_embedding=copy.deepcopy(pretrained))
        # pretrained_embedding=pre_model.layers[0])
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        models.train_model(model, X_train, preprocessing.label_binarize(y_train, classes=labels),
                           nb_epoch=nb_epoch,
                           evaluate=evaluate, max_words=max_words)
        if evaluate is False:
            X_test = models.pad(X_test, max_words=max_words)
        y_test_pred = model.predict_classes(X_test)
        y_test_pred = [labels[i] for i in y_test_pred]
        cm = confusion_matrix(y_test, y_test_pred, labels=labels)
        print(", ".join(labels))
        print("confusion matrix:")
        print(cm)
        scores = precision_recall_fscore_support(y_test, y_test_pred, average=None, labels=labels)
        print("precision, recall, fscore and support values for each class:")
        print(", ".join(labels))
        for x, label in enumerate(["precision", "recall", "fscore", "support"]):
            print(label, scores[x])
            for j, k in enumerate(scores[x]):
                avg_scores[x][j] += k
        print(", ".join(labels))
        cPickle.dump(cm, open("data/scores/%s_cm_cross_%d.pkl" % (task, i), "w"))
        cPickle.dump(cm, open("data/scores/%s_precision_recall_fscore_support_pos_cross_%d.pkl" % (task, i), "w"))


def remap(arr, labels):
    new_arr = np.array(["-------"] * len(arr))
    for ix, k in enumerate(arr):
        for i, j in enumerate(k):
            if j == 1:
                new_arr[ix] = labels[i]
    return new_arr
