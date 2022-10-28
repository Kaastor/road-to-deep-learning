import matplotlib.pyplot as plt


def show_plots(history):
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
