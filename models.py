from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Merge
from keras.regularizers import l2


def Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
    print "Creating text model..."
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim,
                        weights=[embedding_matrix], input_length=seq_length, trainable=False))
    model.add(LSTM(units=512, return_sequences=True, input_shape=(seq_length, embedding_dim)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=512, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1024, activation='tanh'))
    return model


def img_model(dropout_rate):
    print "Creating image model..."
    model = Sequential()
    model.add(Dense(1024, input_dim=4096, activation='tanh'))
    return model


def vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):
    vgg_model = img_model(dropout_rate)
    lstm_model = Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate)
    print "Merging final model..."
    fc_model = Sequential()
    fc_model.add(Merge([vgg_model, lstm_model], mode='mul'))

    # Our code here
    UNITS = 512
    REGULARIZER = 1e-8
    LSTM_BLOCKS = 1

    lstm_decoder = LSTM(units=UNITS, bias_regularizer=l2(REGULARIZER), recurrent_regularizer=l2(REGULARIZER),
                        kernel_regularizer=l2(REGULARIZER), return_sequences=True, name='lstm_decoder')
    fc_model.add(lstm_decoder(LSTM_BLOCKS))
    # End
    # fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(1000, activation='softmax'))
    fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return fc_model

