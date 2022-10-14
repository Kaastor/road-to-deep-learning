import numpy as np
import string

from keras.preprocessing.text import Tokenizer


def word_level_one_hot():
    samples = ['This is a sample sentence number one.', 'I have no patience for more words.']
    token_index = {}
    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                token_index[word] = len(token_index)

    max_len = 10

    results = np.zeros(shape=(
        len(samples), max_len, len(token_index.values())
    ))  # sample, word, index

    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_len]:  # consider first max_len words
            index = token_index.get(word)
            results[i, j, index] = 1
    print(results)


def character_level_one_hot():
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']
    characters = string.printable
    token_index = dict(zip(range(1, len(characters) + 1), characters))
    max_length = 50
    results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))
    for i, sample in enumerate(samples):
        for j, character in enumerate(sample):
            index = token_index.get(character)
            results[i, j, index] = 1


def keras_word_level_one_hot():
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']
    tokenizer = Tokenizer(num_words=1000)  # take into account 1000 most common words
    tokenizer.fit_on_texts(samples)
    sequences = tokenizer.texts_to_sequences(samples)
    one_hot = tokenizer.texts_to_matrix(samples, mode='binary')

    word_index = tokenizer.word_index
    print(f'Found {word_index} unique tokens')


def keras_character_level_one_hot():
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']
    tokenizer = Tokenizer(num_words=None, char_level=True)
    tokenizer.fit_on_texts(samples)
    sequences = tokenizer.texts_to_sequences(samples)
    one_hot = tokenizer.texts_to_matrix(samples, mode='binary')

    word_index = tokenizer.word_index
    print(f'Found {word_index} unique tokens')

