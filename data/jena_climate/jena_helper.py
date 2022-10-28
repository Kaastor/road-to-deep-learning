import numpy as np


def parse_data():
    f = open('/home/przemek/Deep Learning/road-to-deep-learning/road-to-deep-learning/data/jena_climate/data/jena_climate_2009_2016.csv')
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]
    # print(header)
    # print(len(lines))

    float_data = np.zeros((len(lines), len(header) - 1))  # do not take date into the account
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values

    return float_data


"""Python generator that yields batches of data from recent past.
   Sample N and N+1 have most of their samples in common. Generate samples on the fly"""


def generator(data, min_index, delay, lookback, steps, max_index, shuffle=False, batch_size=128):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while True:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:  # end of the data
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))  # handle end of the data
            i += len(rows)
        samples = np.zeros((
            len(rows),
            lookback // steps,
            data.shape[-1]
        ))
        targets = np.zeros((
            len(rows),
        ))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], steps)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]

        yield samples, targets
