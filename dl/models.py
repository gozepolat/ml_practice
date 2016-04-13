from keras import models, layers, callbacks
from keras.preprocessing import sequence


# TODO stride
def construct_pre_model(border_mode='valid', activation='relu',
                        optimizer='adam',
                        lstm_output_size=70, pool_length=2, nb_filter=96,
                        filter_length=3,
                        embedding_size=256, max_words=100, max_features=6000):
    model = models.Sequential()
    model.add(layers.Embedding(max_features, embedding_size, input_length=max_words))
    model.add(layers.core.Dropout(0.5))
    model.add(layers.convolutional.Convolution1D(nb_filter=nb_filter,
                                                 filter_length=filter_length,
                                                 border_mode=border_mode,
                                                 activation=activation,
                                                 subsample_length=1))
    model.add(layers.convolutional.MaxPooling1D(pool_length=pool_length))
    # model.add(layers.core.Dropout(0.1))
    model.add(layers.recurrent.LSTM(lstm_output_size))
    model.add(layers.core.Dense(1))
    model.add(layers.core.Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, class_mode='binary')  # , metrics=["accuracy"])
    # class_mode='binary')
    return model


def construct_cnn_lstm(stateful=False, convolutional=True, loss='categorical_crossentropy', border_mode='valid',
                       activation='relu', optimizer='rmsprop', nb_class=5, lstm_output_size=70, pool_length=2,
                       nb_filter=96, filter_length=3, pretrained_embedding=None, embedding_size=256, max_words=100,
                       max_features=6000, dropouts=[0.5, 0.5, 0.5]):
    model = models.Sequential()
    if pretrained_embedding is None:
        model.add(layers.Embedding(max_features, embedding_size, input_length=max_words))
    else:
        model.add(layers.Embedding(max_features, embedding_size, input_length=max_words,
                                   weights=pretrained_embedding.get_weights()))
    model.add(layers.core.Dropout(dropouts[0]))
    if convolutional:
        model.add(layers.convolutional.Convolution1D(nb_filter=nb_filter,
                                                     filter_length=filter_length,
                                                     border_mode=border_mode,
                                                     activation=activation,
                                                     subsample_length=1))
        model.add(layers.convolutional.MaxPooling1D(pool_length=pool_length))
        model.add(layers.core.Dropout(dropouts[1]))
    if stateful:  # TODO should give input shape if stateful
        model.add(layers.recurrent.LSTM(lstm_output_size, return_sequences=True, stateful=True))
        # ,batch_input_shape=(30, max_words, max_features)))
        model.add(layers.core.Dropout(dropouts[2]))
        model.add(layers.recurrent.LSTM(lstm_output_size, return_sequences=False, stateful=True))
    else:
        model.add(layers.recurrent.LSTM(lstm_output_size))
    model.add(layers.core.Dropout(dropouts[2]))
    model.add(layers.core.Dense(nb_class))
    model.add(layers.core.Activation('softmax'))

    model.compile(loss=loss,
                  optimizer=optimizer)  # , metrics=["accuracy"])
    return model


def pad(x, max_words=100):
    return sequence.pad_sequences(x, maxlen=max_words)


def train_model(model, X_train, y_train, X_test=None, y_test=None, max_words=100, nb_epoch=2, batch_size=30,
                evaluate=False):
    X_train = sequence.pad_sequences(X_train, maxlen=max_words)
    if X_test is not None and y_test is not None:
        X_test = sequence.pad_sequences(X_test, maxlen=max_words)
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  validation_data=(X_test, y_test))
    else:
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch)
    if evaluate:
        score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    else:
        score, acc = None, None
    return score, acc
