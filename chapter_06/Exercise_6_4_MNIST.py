# Adapted from the course template.
# Saved figures and measured results are documented in the matching PDF.

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


# ------------------------
# --- Prepare the images and labels ---
# ------------------------
SEED = 640
keras.utils.set_random_seed(SEED)
(x_all, y_all), (x_test_images, y_test) = keras.datasets.mnist.load_data()
order = np.random.default_rng(SEED).permutation(len(y_all))
train, val = order[6000:], order[:6000]

def flatten(images):
    return images.reshape(-1, 784).astype('float32') / 255.0

x_train = flatten(x_all[train])
x_val = flatten(x_all[val])
x_test = flatten(x_test_images)
y_train = keras.utils.to_categorical(y_all[train], 10)
y_val = keras.utils.to_categorical(y_all[val], 10)
y_test_hot = keras.utils.to_categorical(y_test, 10)
print('Train, validation, test:', len(x_train), len(x_val), len(x_test))


# ------------------------
# --- Define and train a network ---
# ------------------------
def fit_model(widths, dropout, rate=0.001, batch=128, activation='relu',
              seed=SEED, evaluate_test=True):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    # Add the hidden layers in the same order as the chosen widths.
    network_layers = [keras.Input(shape=(784,))]
    for layer_number, width in enumerate(widths):
        network_layers.append(keras.layers.Dense(width, activation=activation))
        if layer_number == 0:
            network_layers.append(keras.layers.Dropout(dropout))
    network_layers.append(keras.layers.Dense(10, activation='softmax'))
    model = keras.Sequential(network_layers)
    model.compile(optimizer=keras.optimizers.Adam(rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch, epochs=10,
                        validation_data=(x_val, y_val), verbose=0).history
    accuracy = None
    if evaluate_test:
        _, accuracy = model.evaluate(x_test, y_test_hot, verbose=0)
    return model, history, accuracy


# ------------------------
# --- Train the 128 node network ---
# ------------------------
baseline, history, baseline_accuracy = fit_model((128,), 0.2)
print(f'Baseline test accuracy: {baseline_accuracy:.2%}')
epochs = np.arange(1, 11)
plt.plot(epochs, history['accuracy'], label='Train')
plt.plot(epochs, history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# ------------------------
# --- Check the confusion matrix ---
# ------------------------
prediction = baseline.predict(x_test, verbose=0).argmax(axis=1)
confusion = np.zeros((10, 10), dtype=int)
np.add.at(confusion, (y_test, prediction), 1)
fig, ax = plt.subplots(figsize=(6, 5))
heatmap = ax.imshow(confusion, cmap='Blues')
ax.set(xlabel='Predicted digit', ylabel='True digit',
       xticks=range(10), yticks=range(10))
for row, col in product(range(10), repeat=2):
    value = confusion[row, col]
    color = 'white' if value > confusion.max() / 2 else 'black'
    ax.text(col, row, str(value), ha='center', va='center', color=color)
fig.colorbar(heatmap, ax=ax)
plt.show()


# ------------------------
# --- Show correct and incorrect predictions ---
# ------------------------
fig, axes = plt.subplots(2, 6, figsize=(10, 4))
for axs, mask in zip(axes, (prediction == y_test, prediction != y_test)):
    for ax, i in zip(axs, np.flatnonzero(mask)[:6]):
        ax.imshow(x_test_images[i], cmap='gray')
        ax.set_title(f'True {y_test[i]} / pred {prediction[i]}')
        ax.axis('off')
fig.tight_layout()
plt.show()


# ------------------------
# --- Compare different hidden layers ---
# ------------------------
print('Architecture, final validation accuracy, test accuracy')
print((128,), f"{history['val_accuracy'][-1]:.2%}",
      f'{baseline_accuracy:.2%}')
for widths in [(64,), (256,), (128, 64), (128, 64, 32, 16)]:
    _, h, test = fit_model(widths, 0.2)
    print(widths, f"{h['val_accuracy'][-1]:.2%}", f'{test:.2%}')


# ------------------------
# --- Grid search over dropout and hidden nodes ---
# ------------------------
widths = [32, 64, 128, 256]
dropouts = [0.0, 0.2, 0.4, 0.6]
grid = []
for dropout, width in product(dropouts, widths):
    _, h, test = fit_model((width,), dropout, rate=0.0001)
    grid.append((width, dropout, h['val_accuracy'][-1], test))
    print(f'{width:3d} units, dropout {dropout:.1f}, '
          f"val {h['val_accuracy'][-1]:.2%}, test {test:.2%}")
print('Grid winner by validation accuracy:', max(grid, key=lambda row: row[2]))
for dropout in dropouts:
    test_by_width = [test for w, d, val, test in grid if d == dropout]
    plt.plot(widths, test_by_width, marker='o', label=f'Dropout {dropout:.1f}')
plt.xlabel('Hidden width')
plt.ylabel('Test accuracy')
plt.legend()
plt.show()


# ------------------------
# --- Random search over five hyperparameters ---
# ------------------------
spaces = [
    [8, 32, 128, 256, 512],
    [0, 0.1, 0.2, 0.3, 0.5, 0.6],
    [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    [16, 32, 64, 128, 256],
    ['relu', 'elu', 'sigmoid', 'tanh'],
]
candidates = list(product(*spaces))
loss_


# ------------------------
# --- Plot the effect of the number of trials ---
# ------------------------
curves = []
for repeat in range(3):
    rng = np.random.default_rng(SEED + repeat)
    picks = rng.choice(len(candidates), 20, replace=False)
    trials = []
    for number, ix in enumerate(picks, 1):
        width, dropout, rate, batch, activation = candidates[ix]
        _, h, test = fit_model(
            (width,), dropout, rate=rate, batch=batch, activation=activation,
            seed=SEED + repeat * 20 + number, evaluate_test=(repeat == 0),
        )
        val_loss = h['val_loss'][-1]
        val_acc = h['val_accuracy'][-1]
        trials.append((candidates[ix], val_loss, val_acc, test))
        if repeat == 0:
            print(f'{number:2d} {candidates[ix]} '
                  f'val loss {val_loss:.4f}; val {val_acc:.2%}; test {test:.2%}')
    if repeat == 0:
        print('Random winner by validation accuracy:',
              max(trials, key=lambda row: row[2]))
    loss_curves.append(np.minimum.accumulate([trial[1] for trial in trials]))

curves = np.array(loss_curves)
mean = curves.mean(axis=0)
sd = curves.std(axis=0, ddof=1)
budget = np.arange(1, 21)
plt.plot(budget, mean, label='Mean best validation loss')
plt.fill_between(budget, mean - sd, mean + sd, alpha=0.25, label='±1 SD')
plt.xlabel('Number of trials')
plt.ylabel('Best validation cross entropy')
plt.legend()
plt.show()
