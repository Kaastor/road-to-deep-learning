import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop

from data.jena_climate.jena_helper import parse_data, generator

tf.keras.backend.clear_session()

lookback = 720  # observations will go back 5 days
steps = 6  # observations will be sampled 1 data point per hour (timestep is 10 minutes in dataset)
delay = 144  # targets will be 24 hours in the future

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

"""2. Use Python generator that yields batches of data from recent past.
   Sample N and N+1 have most of their samples in common. Generate samples on the fly"""

train_gen = generator(dataset, delay=delay, lookback=lookback, steps=steps, min_index=0, max_index=200000, shuffle=True)
val_gen = generator(dataset, delay=delay, lookback=lookback, steps=steps, min_index=200001, max_index=300000, shuffle=True)
test_gen = generator(dataset, delay=delay, lookback=lookback, steps=steps, min_index=300000, max_index=None, shuffle=True)

val_steps = (300000 - 200001 - lookback)  # length of val dataset, number of timesteps
test_steps = len(dataset) - 300001 - lookback  # length of test dataset, number of timesteps

"""3. Common-sense baseline - temp should be the same in 24 hours. Use MAE metric."""


def eval_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))  # 0.29
    celcius_mae = np.mean(batch_maes) * std[1]  # ~2.57 C


# eval_naive_method()

"""
4. Basic ML approach - try simple model to see if more complex models are necessary.
It turns out common sense is not easy to outperform. Your common sense contains a lot of valuable information
that a machine-learning model doesn’t have access to.

Result: 
"""


def basic_model():
    return Sequential([
        # removes the notion of time from the series
        layers.Flatten(input_shape=(lookback // steps, dataset.shape[-1])),  # (120, 14)
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])


"""
5. RNN approach - GRU, cheaper than LSTM, but do not have as much representational
power.
Networks being regularized with dropout always take longer to fully converge, 
train the network for twice as many epochs.

Next steps:
- adjust number of units in RNN layers
- adjust learning rate
- use LSTM instead of GRU
- use bigger regressor (more Dense layers, bigger ones)
- run on the test set!
"""


def gru_simple_model():
    return Sequential([
        layers.GRU(32,
                   dropout=0.2,  # rate for input units
                   recurrent_dropout=0.2,  # rate for recurrent units
                   input_shape=(None, dataset.shape[-1])),
        layers.Dense(1)
    ])


# increased capacity of the network, increase the
# capacity of your network until overfitting becomes the primary obstacle
def gru_large_model():
    return Sequential([
        layers.GRU(32,
                   dropout=0.1,  # rate for input units
                   recurrent_dropout=0.5,  # rate for recurrent units
                   input_shape=(None, dataset.shape[-1]),
                   return_sequences=True),
        layers.GRU(64, activation='relu',
                   dropout=0.1,
                   recurrent_dropout=0.5),
        layers.Dense(1)
    ])


"""Training the model."""
model = gru_simple_model()
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit(train_gen,
                    steps_per_epoch=500,
                    epochs=20,
                    validation_data=val_gen,
                    validation_steps=val_steps)

"""Show results."""
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
