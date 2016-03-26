from keras import models, layers, callbacks
from keras.preprocessing import sequence


def construct_cnn_lstm(stateful=False, convolutional=True, loss='categorical_crossentropy', border_mode='valid', activation='relu',
                       optimizer='rmsprop', nb_class=5,
                       lstm_output_size=70, pool_length=2, nb_filter=64,
                       filter_length=3,
                       embedding_size=128, max_words_in_sentence=100, max_features=20000, dropouts=[0.25, 0.1, 0.2]):
    model = models.Sequential()
    model.add(layers.Embedding(max_features, embedding_size, input_length=max_words_in_sentence))
    #model.add(layers.core.Dropout(dropouts[0]))
    if convolutional:
        model.add(layers.convolutional.Convolution1D(nb_filter=nb_filter,
                                                 filter_length=filter_length,
                                                 border_mode=border_mode,
                                                 activation=activation,
                                                 subsample_length=1))
        model.add(layers.convolutional.MaxPooling1D(pool_length=pool_length))
        model.add(layers.core.Dropout(dropouts[1]))
    if stateful:
        model.add(layers.recurrent.LSTM(lstm_output_size, return_sequences=True, stateful=True))
        model.add(layers.core.Dropout(dropouts[2]))
        model.add(layers.recurrent.LSTM(lstm_output_size, return_sequences=False, stateful=True))
    else:
        model.add(layers.recurrent.LSTM(lstm_output_size))
    model.add(layers.core.Dropout(dropouts[2]))
    model.add(layers.core.Dense(nb_class))
    model.add(layers.core.Activation('softmax'))

    model.compile(loss=loss,
                  optimizer=optimizer)
    return model


def train_model(model, X_train, X_test, y_train, y_test, max_words_in_sentence=100, nb_epoch=2, batch_size=30,
                patience=2):
    X_train = sequence.pad_sequences(X_train, maxlen=max_words_in_sentence)
    X_test = sequence.pad_sequences(X_test, maxlen=max_words_in_sentence)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(X_test, y_test), show_accuracy=True,) # callbacks=[callbacks.EarlyStopping(monitor='val_acc', patience=patience, verbose=1, mode='auto')]
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size,
                                show_accuracy=True)
    return score, acc
