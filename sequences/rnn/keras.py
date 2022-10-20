from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Bidirectional, Embedding, Dense, LSTM
from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt

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


print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=max_len_of_words_in_seq)
input_test = sequence.pad_sequences(input_test, maxlen=max_len_of_words_in_seq)
print('input_train shape:', input_train.shape)

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

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='training acc')
plt.plot(epochs, val_acc, 'b', label='val acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='val loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
