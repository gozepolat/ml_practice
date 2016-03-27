import os
from dl import models



import cPickle

"""from sklearn import cross_validation

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, Imputer
"""

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
    (pre_y, pre_X, pos_neg_y, pos_neg_X, pos_X_train, pos_X_test, pos_y_train, pos_y_test, neg_X_train, neg_X_test,
     neg_y_train, neg_y_test, max_words_in_sentence) = cPickle.load(open("data/all.pkl","r"))


    def train(initial_model, X_train, X_test, y_train, y_test, nb_epoch=16,
              batch_size=30, max_words=max_words_in_sentence,evaluate=True):
        return models.train_model(initial_model, X_train, X_test, y_train, y_test, nb_epoch=nb_epoch,
                                  batch_size=batch_size, max_words_in_sentence=max_words, evaluate=evaluate)
    # pretraining
    pre_model = models.construct_pre_model(max_words_in_sentence=max_words_in_sentence)

    if not os.path.isfile("data/pos_neg_model.hdf5"):
        if not os.path.isfile("data/pre_model.hdf5"):  # train valid/invalid model
            split_ix = int(len(pre_X) * (1 - 0.2))
            score, acc = train(pre_model, pre_X[:split_ix], pre_X[split_ix:], pre_y[:split_ix], pre_y[split_ix:], nb_epoch=1)
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
        print("end of pretraining.. loading the pretrained embedding layer weights")
        pre_model.load_weights("data/pos_neg_model.hdf5")
    #print("using the word embeddings learned from pretraining for classification tasks..")
    #skf = StratifiedKFold(labels, n_folds=n_folds, shuffle=True)
    #print("starting cross-validation")


    # train the dataset on a cnn / lstm architecture
    pos_model = models.construct_cnn_lstm(nb_class=4, stateful=False, convolutional=True,
                                          max_words_in_sentence=max_words_in_sentence,
                                          pretrained_embedding=pre_model.layers[0])
    score, acc = train(pos_model, pos_X_train, pos_X_test, pos_y_train, pos_y_test, nb_epoch=30)

    # use the same embedding layer for neg?
    print("positive model is trained with validation loss and accuracy", (score, acc))

    neg_model = models.construct_cnn_lstm(nb_class=5, stateful=False, convolutional=True,
                                          max_words_in_sentence=max_words_in_sentence,
                                          pretrained_embedding=pre_model.layers[0])
    score, acc = models.train_model(neg_model, neg_X_train, neg_X_test, neg_y_train, neg_y_test, nb_epoch=30)

    # use the same embedding layer for neg?
    print("negative model is trained with validation loss and accuracy", (score, acc))

    from sklearn.cross_validation import StratifiedKFold
    # cross-validation
    n_folds = 10
    # pos model
    pos_y=pos_y_train+pos_y_test
    pos_X=pos_X_train+pos_X_test
    skf = StratifiedKFold(pos_y, n_folds=n_folds, shuffle=False, random_state=None)  # already shuffled
    # precision, recall, f1-score and support on the test data for each class should be recorded and provided, as well as the confusion matrix.
    for i, (train_ix, test_ix) in enumerate(skf):
            print ("Cross-validation fold: %d/%d"%(i+1, n_folds))
            pos_model = None # Clearing the NN.
            pos_model = models.construct_cnn_lstm(nb_class=4, stateful=False, convolutional=True,
                                          max_words_in_sentence=max_words_in_sentence,
                                          pretrained_embedding=pre_model.layers[0])
            X_train, X_test = pos_X[train_ix], pos_X[test_ix]
            y_train, y_test = pos_y[train_ix], pos_y[test_ix]
            train(model, data[train], labels[train], data[test], labels[test),evaluate=False)





"""
def load_data():
    # load your data using this function

def create model():
    # create your model using this function

def train_and_evaluate__model(model, data[train], labels[train], data[test], labels[test)):
    model.fit...
    # fit and evaluate here.

# 10-fold stratified cross-validation
if __name__ == "__main__":
    n_folds = 10
    data, labels, header_info = load_data()
    skf = StratifiedKFold(labels, n_folds=n_folds, shuffle=True)

    for i, (train, test) in enumerate(skf):
            print ("Cross-validation fold: %d/%d"%(i+1, n_folds))
            model = None # Clearing the NN.
            model = construct_cnn_lstm()
            train_and_evaluate_model(model, data[train], labels[train], data[test], labels[test))

"""
