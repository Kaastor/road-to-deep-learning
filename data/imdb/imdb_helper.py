from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence


def get_data(max_features, max_len_of_words_in_seq):
    print('Loading data...')
    (input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(input_train), 'train sequences')
    print(len(input_test), 'test sequences')

    print('Pad sequences (samples x time)')
    input_train = sequence.pad_sequences(input_train, maxlen=max_len_of_words_in_seq)
    input_test = sequence.pad_sequences(input_test, maxlen=max_len_of_words_in_seq)
    print('input_train shape:', input_train.shape)

    return (input_train, y_train), (input_test, y_test)
