# Adapted from the course template.
# Saved figures and measured results are documented in the matching PDF.

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


# ------------------------
# --- Prepare the training and test examples ---
# ------------------------
seed = 620
keras.utils.set_random_seed(seed)
rng = np.random.default_rng(seed)

def target(x):
    return (np.sqrt(np.abs(x)) + 0.1*x + 0.01*x**2 + 1
            - np.sin(x) + 0.5*np.exp(x/10)) / (0.5 + np.abs(np.cos(x)))

x_train = rng.uniform(-10, 10, (10_000, 1)).astype('float32')
y_train = (target(x_train) + rng.normal(size=x_train.shape)).astype('float32')
x_test = np.linspace(-10, 10, 10_000, dtype='float32').reshape(-1, 1)
y_test = target(x_test)


# ------------------------
# --- Define and compile the network ---
# ------------------------
model = keras.Sequential(
    [keras.Input(shape=(1,))]
    + [keras.layers.Dense(100, activation='relu') for _ in range(8)]
    + [keras.layers.Dense(1)]
)
model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse', steps_per_execution=32)
model.summary(line_length=75)

class RecordPredictions(keras.callbacks.Callback):
    def __init__(self, inputs):
        super().__init__()
        self.inputs = inputs
        self.predictions = {}

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 20 == 0:
            self.predictions[epoch + 1] = self.model.predict(
                self.inputs, verbose=0
            ).ravel()


# ------------------------
# --- Train the network and save predictions ---
# ------------------------
snapshots = RecordPredictions(x_test)
history = model.fit(
    x_train, y_train, epochs=200, batch_size=64, callbacks=[snapshots], verbose=0
)
final_prediction = snapshots.predictions[200]
test_mse = np.mean((final_prediction - y_test.ravel())**2)
print(f"Final training MSE: {history.history['loss'][-1]:.4f}")
print(f"Noiseless test MSE: {test_mse:.4f}")


# ------------------------
# --- Plot the predictions and training loss ---
# ------------------------
fig, (prediction_ax, loss_ax) = plt.subplots(2, 1, figsize=(10, 8))
prediction_ax.plot(x_test.ravel(), y_test.ravel(), 'k', label='Exact target')
for epoch, predictions in snapshots.predictions.items():
    prediction_ax.plot(x_test.ravel(), predictions, label=str(epoch), alpha=0.8)
prediction_ax.set(xlabel='x', ylabel='Function value')
prediction_ax.legend(title='Epoch', ncol=3)
loss_ax.semilogy(np.arange(1, 201), history.history['loss'])
loss_ax.set(xlabel='Epoch', ylabel='Training MSE')
fig.tight_layout()
plt.show()
