# Adapted from the course exercise template.

from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# Load and prepare the wine data.
path = Path('winequality-white.csv')
if not path.exists():
    urlretrieve(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/'
        'wine-quality/winequality-white.csv',
        path,
    )
data = np.genfromtxt(path, delimiter=';', skip_header=1)
np.random.default_rng(1234).shuffle(data)
train, validation, test = data[:2550], data[2550:3000], data[3000:]
mean = train[:, :11].mean(axis=0)
std = train[:, :11].std(axis=0)

def split(rows):
    return ((rows[:, :11] - mean) / std).astype('float32'), rows[:, 11]

x_train, y_train = split(train)
x_val, y_val = split(validation)
x_test, y_test = split(test)

# Define the network.
def build_model(widths, optimizer):
    keras.utils.set_random_seed(630)
    model = keras.Sequential(
        [keras.Input(shape=(11,))]
        + [keras.layers.Dense(width, activation='relu') for width in widths]
        + [keras.layers.Dense(1)]
    )
    model.compile(optimizer=optimizer, loss='mse', steps_per_execution=32)
    return model

# Choose the optimizer and network settings.
configs = [
    ('SGD', (20,), 'sgd', 1),
    ('Momentum SGD', (20,), 'momentum', 1),
    ('Adam', (20,), 'adam', 1),
    ('Adam, deeper', (64, 32, 16), 'adam', 1),
    ('Adam, deeper, batch 5', (64, 32, 16), 'adam', 5),
    ('Adam, compact, batch 5', (32, 16, 8), 'adam', 5),
]
optimizers = {
    'sgd': lambda: keras.optimizers.SGD(learning_rate=0.01),
    'momentum': lambda: keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'adam': lambda: keras.optimizers.Adam(learning_rate=0.001),
}

# Train each setting.
runs = {}
for label, widths, optimizer, batch in configs:
    model = build_model(widths, optimizers[optimizer]())
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=30, batch_size=batch, verbose=0,
    ).history
    val_prediction = model.predict(x_val, verbose=0).ravel()
    val_r = np.corrcoef(val_prediction, y_val)[0, 1]
    runs[label] = (model, history, val_r)

# Compare the results.
winner = max(runs, key=lambda label: runs[label][2])
print(f'Selected by validation correlation: {winner}')
print(f"{'Model':30} {'Val r':>8} {'Test MSE':>10} {'Test r':>8}")
for label, (model, _, val_r) in runs.items():
    prediction = model.predict(x_test, verbose=0).ravel()
    test_mse = np.mean((prediction - y_test)**2)
    test_r = np.corrcoef(prediction, y_test)[0, 1]
    print(f'{label:30} {val_r:8.4f} {test_mse:10.4f} {test_r:8.4f}')

# Plot the loss and predictions.
fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
for ax, (label, (_, history, _)) in zip(axes.flat, runs.items()):
    epochs = np.arange(1, len(history['loss']) + 1)
    ax.semilogy(epochs, history['loss'], label='Train')
    ax.semilogy(epochs, history['val_loss'], label='Validation')
    ax.set(title=label, xlabel='Epoch', ylabel='MSE')
    ax.legend()
fig.tight_layout()
plt.show()

selected_model = runs[winner][0]
prediction = selected_model.predict(x_test, verbose=0).ravel()
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_test, prediction, s=12, alpha=0.25)
lo = min(y_test.min(), prediction.min())
hi = max(y_test.max(), prediction.max())
ax.plot([lo, hi], [lo, hi], 'k--', label='Exact prediction')
ax.set(xlabel='Observed quality', ylabel='Predicted quality', title=winner)
ax.legend()
plt.show()
