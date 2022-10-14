import os
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Embedding, Flatten, Dense
from keras import Sequential
from keras.datasets import imdb
from keras import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

maxlen = 100  # Take only first 100 words from the review
training_samples = 1000  # train model on 200 samples
validation_samples = 10000  # validate on 10000 samples
max_words = 10000  # take into the consideration only top words in the dataset
embedding_dim = 100


def download_imdb_train_data():
    # link to data: http://mng.bz/0tIo
    imdb_dir = './data/aclImdb'
    train_dir = os.path.join(imdb_dir, 'train')

    labels = []
    texts = []

    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)

    return texts, labels


def download_imdb_test_data():
    # link to data: http://mng.bz/0tIo
    imdb_dir = './data/aclImdb'
    test_dir = os.path.join(imdb_dir, 'test')

    labels = []
    texts = []

    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(test_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)

    return texts, labels


def tokenize_raw_imdb(train_texts, test_texts, train_labels, test_labels):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train_texts)
    seq_train = tokenizer.texts_to_sequences(train_texts)
    seq_test = tokenizer.texts_to_sequences(test_texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data_train = pad_sequences(seq_train, maxlen=maxlen)  # pads and truncates the sequences to maxlen
    x_test = pad_sequences(seq_test, maxlen=maxlen)  # pads and truncates the sequences to maxlen

    train_labels = np.asarray(train_labels)
    y_test = np.asarray(test_labels)
    print('Shape of data tensor:', data_train.shape)
    print('Shape of label tensor:', train_labels.shape)

    indices = np.arange(data_train.shape[0])  # create an array of values: [0, 1, ..., data.shape[0]=num_of_examples]
    np.random.shuffle(indices)
    data_train = data_train[indices]  # https://www.geeksforgeeks.org/indexing-in-numpy/, returns values at the specified indices
    train_labels = train_labels[indices]
    x_train = data_train[:training_samples]
    y_train = train_labels[:training_samples]
    x_val = data_train[training_samples: training_samples + validation_samples]
    y_val = train_labels[training_samples: training_samples + validation_samples]

    return x_train, y_train, x_val, y_val, x_test, y_test, word_index


# 100-dimensional embedding vectors for 400,000 words (or non-word tokens)
def parse_glove_embeddings_file():
    glove_dir = './GloVe'
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]  # first in the line is the word
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def prepare_word_embedding_matrix(word_index):
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vec = embeddings_index.get(word)
            if embedding_vec is not None:
                embedding_matrix[i] = embedding_vec

    return embedding_matrix


def get_data(maxlen, max_features):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

    return (x_train, y_train), (x_test, y_test)


def plot_results(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def embedding_layer_imdb(embedding_matrix, x_train, y_train, x_val, y_val, x_test, y_test):
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))  # 1000 possible tokens, dim of embeddings
    # output: (num_samples, maxlen, dim)
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # un comment to use pretrained embeddings
    # model.layers[0].set_weights([embedding_matrix])
    # model.layers[0].trainable = False  # freeze the layer

    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['acc']
    )
    model.summary()

    history = model.fit(x_train,
                        y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(x_val, y_val))
    model.save_weights('pre_trained_glove_model.h5')
    plot_results(history)

    # evaluate model
    print('Model evaluation:')
    model.load_weights('pre_trained_glove_model.h5')
    model.evaluate(x_test, y_test)


train_texts, train_labels = download_imdb_train_data()
test_texts, test_labels = download_imdb_test_data()
x_train, y_train, x_val, y_val, x_test, y_test, word_index = tokenize_raw_imdb(
    train_texts=train_texts,
    test_texts=test_texts,
    train_labels=train_labels,
    test_labels=test_labels
)
embeddings_index = parse_glove_embeddings_file()
emb_matrix = prepare_word_embedding_matrix(word_index)
embedding_layer_imdb(emb_matrix, x_train, y_train, x_val, y_val, x_test, y_test)

