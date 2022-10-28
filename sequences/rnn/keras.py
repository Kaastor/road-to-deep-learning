import matplotlib.pyplot as plt
from tensorflow.keras.layers import SimpleRNN, Bidirectional, Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential

from data.imdb.imdb_helper import get_data
from helpers.keras_vis import show_plots

max_features = 10000  # max number of most frequent words used
max_len_of_words_in_seq = 500
batch_size = 32


def get_simple_model():
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    return model


def get_lstm_model():
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    return model


"""
Bidirectional creates a second, separate instance of this recurrent
layer and uses one instance for processing the input sequences in 
reversed order.
~96% val accuracy.
"""


def get_bidirectional_lstm():
    return Sequential([
        Embedding(max_features, 32),
        Bidirectional(LSTM(32, dropout=0.1, recurrent_dropout=0.4)),
        Dense(1, activation='sigmoid')
    ])


def get_model():
    model = Sequential()
    model.add(Embedding(10000, 32))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32, return_sequences=True))  # stacking increase representational power
    model.add(SimpleRNN(32))  # only last layer returns last output
    model.summary()

    return model


(input_train, y_train), (input_test, y_test) = get_data(max_features, max_len_of_words_in_seq)

model = get_bidirectional_lstm()
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)
history = model.fit(
    input_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)

show_plots(history)
