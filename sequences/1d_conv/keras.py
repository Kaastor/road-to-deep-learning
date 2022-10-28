from data.imdb.imdb_helper import get_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPool1D, Conv1D, Embedding, GlobalMaxPooling1D, Dense
from tensorflow.keras.optimizers import RMSprop

from helpers.keras_vis import show_plots

max_features = 10000  # max number of most frequent words used
max_len_of_words_in_seq = 500
batch_size = 32
embedding_dim = 128

(input_train, y_train), (input_test, y_test) = get_data(max_features, max_len_of_words_in_seq)


def conv_1D_model():
    return Sequential([
        Embedding(max_features, embedding_dim, input_length=max_len_of_words_in_seq),
        Conv1D(32, 7, activation='relu'),
        MaxPool1D(5),
        Conv1D(32, 7, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(1, activation='sigmoid')
    ])


model = conv_1D_model()
model.summary()

model.compile(optimizer=RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

show_plots(history)
