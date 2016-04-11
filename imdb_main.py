import os
from dl import models
import cPickle

from prep import dataset

# nltk.download("punkt")
# nltk.download("words")
# nltk.download("stopwords")
# nltk.download("twitter_samples")


if __name__ == "__main__":
    if not os.path.isfile("data/imdb.pkl"):
        (pre_y, pre_X, X_tr, X_test, y_tr, y_test, test, max_words_in_sentence) = dataset.dump_imdb()
        pass
    else:
        print("loading preprocessed data..")
        (pre_y, pre_X, X_tr, X_test, y_tr, y_test, test, max_words_in_sentence) = cPickle.load(
            open("data/imdb.pkl", "r"))
    print("data is ready for training!")
    word_int_map=cPickle.load(open("data/imdb_word_int_map.pkl"))
    nb_features = len(word_int_map.left_to_right) + 5
    print(nb_features)
    pre_model = models.construct_pre_model(max_words=max_words_in_sentence, max_features=nb_features)
    print("constructed initial model")
    if not os.path.isfile("data/imdb_pre_model.hdf5"):  # train valid/invalid model
        split_ix = int(len(pre_X) * (1 - 0.2))
        score, acc = models.train_model(pre_model, pre_X[:split_ix], pre_y[:split_ix], pre_X[split_ix:],
                                        pre_y[split_ix:],
                                        nb_epoch=1)
        pre_model.save_weights("data/pre_model.hdf5", overwrite=True)
        print("(pre)training of valid/invalid set is completed with validation loss and accuracy", (score, acc))
    else:
        # print("loading the pretrained embedding layer weights")
        pre_model.load_weights("data/imdb_pre_model.hdf5")

    imdb_model = models.construct_cnn_lstm(nb_class=5, stateful=False, convolutional=True,
                                           max_words=max_words_in_sentence, max_features=nb_features,
                                           pretrained_embedding=pre_model.layers[0])
    score, acc = models.train_model(imdb_model, X_tr, y_tr, X_test, y_test, nb_epoch=40,
                                    max_words=max_words_in_sentence)
    # predict:
    test = models.pad(test, max_words=max_words_in_sentence)
    pred = imdb_model.predict_classes(test)
    pos_labels = ["0", "1", "2", "3", "4"]
    pred_labels = [pos_labels[i] for i in pred]

    imdb_model.save_weights("data/pos_model.hdf5", overwrite=True)
    with open("data/test.tsv", "r") as test, open(
            "data/pred.csv", "w") as pred:
        i = 0
        header = next(test)
        for line in test:
            pred.write(line.strip() + pred_labels[i] + "\n")
            i += 1
