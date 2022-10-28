from data.jena_climate.jena_helper import generator, parse_data
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop

from helpers.keras_vis import show_plots

"""
Using smaller step means higher resolution timeseries
"""
step = 3
lookback = 720
delay = 144
training_timesteps = 200000

dataset = parse_data()
temp = dataset[:, 1]  # take only first column
# data is recorded every 10 minutes, 144 data points per day
plt.plot(range(1440), temp[:1440])  # data for 10 days
# plt.show()

"""1. Scaling the data to take small values for each timeseries"""
# z-score normalization for training data
mean = dataset[:training_timesteps].mean(axis=0)
dataset = dataset - mean
std = dataset[:training_timesteps].std(axis=0)
dataset = dataset / std

train_gen = generator(dataset,
                      lookback=lookback,
                      delay=delay,
                      steps=step,
                      min_index=0,
                      max_index=200000,
                      shuffle=True)
val_gen = generator(dataset,
                    lookback=lookback,
                    delay=delay,
                    steps=step,
                    min_index=200001,
                    max_index=300000)
test_gen = generator(dataset,
                     lookback=lookback,
                     delay=delay,
                     steps=step,
                     min_index=300001,
                     max_index=None)
val_steps = (300000 - 200001 - lookback) // 128
test_steps = (len(dataset) - 300001 - lookback) // 128

"""
Prepare RNN-CNN model to combine speed and lightness of convnets with
order-sensitivity of RNNs is to use a 1D convnet as a preprocessing step before an RNN.
Good for very long sequences that can't be realistically processed by RNN. 
Convnet will turn long input sequence into much shorter, downsampled sequences of 
high level features. This sequence of extracted features becomes input to RNN part
of the network.
"""


def rnn_cnn_model():
    return Sequential([
        layers.Conv1D(32, 9,
                      activation='relu',
                      input_shape=(None, dataset.shape[-1])),
        layers.MaxPool1D(3),
        layers.Conv1D(32, 5, activation='relu'),
        layers.MaxPool1D(3),
        layers.GRU(32, dropout=0.1, recurrent_dropout=0.5),
        layers.Dense(1)
    ])


model = rnn_cnn_model()
model.summary()

model.compile(optimizer=RMSprop(), loss='mae', metrics=['acc', 'val_acc'])
history = model.fit(train_gen,
                    steps_per_epoch=500,
                    epochs=20,
                    validation_data=val_gen,
                    validation_steps=val_steps)

show_plots(history)
