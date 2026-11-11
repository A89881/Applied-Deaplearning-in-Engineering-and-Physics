# Applied deep learning exercise walkthrough

Read each section alongside its notebook. The code cells are counted from the top, skipping assignment text. Blank lines group related calculations. The last code cell in each notebook has a short conclusion.

## Exercise 4.1 — Linear regression

### Code cell 1 · Imports and plotting

- `import numpy as np` loads the array and linear algebra tools. `import matplotlib.pyplot as plt` loads the plotting tools.
- `plt.style.use(...)` sets the plot styling. `plt.rcParams.update(...)` selects a default color map for plots that use one. These lines only affect appearance.

### Code cells 2–3 · Load and split the wine data

- `Path("winequality-white.csv")` points to the CSV in the working directory. `is_file()` checks whether it is already there. If the file is missing, `urlretrieve(..., data_path)` downloads it from UCI and saves it there.
- `np.genfromtxt(..., delimiter=";", skip_header=1)` reads the numbers into a matrix. The CSV uses semicolons between fields. The header is a row of names, so we skip it. `data.shape` prints `(4898, 12)`: 4,898 rows, 11 measurements, and one score.
- `np.random.default_rng(1234)` makes the shuffle repeatable. `rng.shuffle(data)` shuffles whole rows, so each wine stays paired with its quality score.
- `data[:3000, :11]` takes the first 3,000 wines and the first 11 columns for `X_train`. `data[:3000, 11]` takes their quality scores for `y_train`. The `3000:` slices put the remaining 1,898 wines in the test set. The print statements let you check that a training example has 11 inputs and one target.

### Code cell 4 · Explore each feature

- `feature_names` follows the CSV column order. `np.zeros(len(feature_names))` allocates one place for each correlation.
- `for i in range(...)` visits the columns. `X_train[:, i]` means all training wines in column `i`. `np.corrcoef(...)[0, 1]` selects the correlation between that column and quality from the 2×2 correlation matrix. A negative result means quality tends to decrease as that feature increases.
- The outer plotting loop starts at indices 0, 4, and 8. `min(...)` stops the last figure at the last feature. `plt.subplots(..., squeeze=False)` keeps the axes in a two-dimensional array, with a histogram and scatter plot per row.
- `zip(indices, axes)` pairs each feature index with its two axes. `hist(..., bins=30)` shows how common different measurement values are. `scatter(feature, y_train, ...)` shows quality against the measurement. A low `alpha` helps reveal overlapping points. The title prints the signed correlation.
- `np.argmax(np.abs(correlations))` finds the position with the largest magnitude, regardless of sign. The final print uses that index to report alcohol, with `r = 0.429`. This ranks the individual linear associations. It does not measure what the feature contributes after the other features are included.

### Code cell 5 · Solve for the weights

- The model predicts `X_train @ weights`, one value per row. The exercise specifies one weight per feature, so the input matrix has no extra column of ones.
- `A = X_train.T @ X_train` forms the 11×11 matrix on the left of the normal equations. `b = X_train.T @ y_train` forms the length-11 vector on the right. Setting the derivative of the squared residual sum to zero gives `A @ weights = b`. This is the same least-squares pattern as the normal-equation example in the 2024 practice notebook.
- `np.linalg.solve(A, b)` solves the system directly. The `zip` loop prints the coefficient next to its feature. Coefficients depend on each feature's units and on the other columns in the model, so their magnitudes answer a different question from the correlations.

### Code cell 6 · Test predictions

- `y_pred = X_test @ weights` applies the same weights to the unseen wines. `corrcoef(y_pred, y_test)[0, 1]` measures how closely changes in the predictions track changes in the true scores.
- `error = y_pred - y_test` gives a signed residual for every test wine. `np.sqrt(np.mean(error**2))` squares, averages, and takes the square root. The resulting RMSE is in quality points. The two print statements report `r = 0.516` and `RMSE = 0.753`.
- The scatter plot puts actual quality on the horizontal axis and predicted quality on the vertical axis. `limits` uses the full range of both arrays. The dashed diagonal shows where a perfect prediction would land. Equal axis scaling lets you compare the distances visually. The integer tick labels come from `np.unique(y_test)`.

## Exercise 4.2 — The checkerboard

### Code cell 1 · Data, features, and the network

- `rng.uniform(-5, 5, size=(500, 2))` gives 500 pairs of independently drawn coordinates. The `points += ...` line shifts each positive coordinate up by 0.3 and each negative one down by 0.3, matching the Playground's gap near the axes. The notebook uses NumPy's random stream, so it draws different points from the browser.
- `points[:, 0] * points[:, 1]` multiplies the two coordinates for every point. `np.where(... >= 0, 1, -1)` assigns one class to equal signs and the other class to opposite signs. The label depends on both coordinates even though the draws themselves are independent. `np.split(..., 2)` gives 250 training and 250 test points.
- `features` holds three transforms. The raw case returns the original array. `np.column_stack` adds columns for the two squares or the product in the other cases. Each setting uses the same underlying points and labels.
- In `forward`, `activations = [x]` stores the input batch. Each hidden layer computes `activations[-1] @ weight + bias`. Matrix multiplication gives a value for each example and unit. The bias broadcasts across rows. `np.maximum(..., 0)` applies ReLU and saves the result for the backward pass.
- The last layer uses `tanh` to produce a score between −1 and 1. The decision boundary is where the score is zero. With no hidden layers, `tanh` preserves the sign of an affine function, so the boundary in the supplied features is a single line or hyperplane.
- In `train`, the seed controls the initial weights and batch order. `transform(train_x)` prepares the chosen features. `sizes` joins the input width, the hidden widths in `shape`, and the single output unit. `zip(sizes, sizes[1:])` gives each pair of adjacent layer widths. Each weight matrix has one row per incoming unit and one column per outgoing unit. Each bias vector has one entry per outgoing unit.
- The epoch loop makes repeated passes over the training points. `rng.permutation(len(x))` shuffles the row indices. `.reshape(-1, 10)` groups those indices into batches of ten. `train_y[batch, None]` gives the targets a column dimension, matching the score shape `(10, 1)`.
- `error` is the score minus the target. For half squared error, the slope with respect to the score is just that difference. `tanh_slope = 1 - score**2` gives the derivative of tanh. `delta` multiplies these two slopes and divides by the batch size, so the weight update uses the mean batch gradient.
- The reverse loop starts at the output. `delta @ weights[layer].T` moves the error back through that layer's connections. `activations[layer] > 0` passes it only through active ReLU units. The previous delta is computed before the weights change. At the first layer, the previous delta would refer to the input features and is unused.
- `dW = activations[layer].T @ delta` combines the input to each connection with the error arriving at its output. `db = delta.sum(axis=0)` adds the bias contributions across examples. The two update lines subtract learning rate times gradient. `delta = previous_delta` prepares the next step back.
- The inner `predict` function reuses the trained weights and the same feature transform. `[:, 0]` takes the single score column as a vector. Returning that function lets the next cell predict on held-out data or on a plotting grid.

### Code cell 2 · Compare architectures and repeat runs

- `shapes` lists the hidden widths. `()` is a model with zero hidden layers, `(3,)` means one layer of width three, and `(4, 2)` means two hidden layers.
- `models` trains once per seed. The accuracy loop converts scores to `1` or `-1` with `np.where`. Comparing that vector with `test_y` gives a Boolean for each point. Its mean is the fraction correct.
- `trials[shape, feature_name]` stores the models and accuracies for later plotting. The table prints the five accuracies and the fraction of runs that reach the chosen 95% threshold.
- `next(..., None)` takes the first listed architecture with at least one successful run. In this list, three units in one layer come first and reached 100% in all five runs. That is an empirical result under this split and training setup. The unfinished second question was treated as a repeatability question. For example, the `(4, 2)` model reached 100% in four runs and 88% in one.

### Code cell 3 · Compare features and draw boundaries

- `compare((), "squares")` and `compare((), "product")` keep the architecture fixed at zero hidden layers, which isolates the effect of the added features.
- `linspace` makes evenly spaced coordinates. `meshgrid` forms all coordinate pairs, and `column_stack` turns them into the same two-column layout as the examples.
- The plot loop retrieves each setting's best run with `argmax(scores)`. The predicted grid is reshaped to match `xx` and `yy`. `contourf` shades the negative and positive decision regions. `scatter` overlays the held-out points with their true colors.
- The product helped most because its sign alone gives the class. The squares have the same values when a coordinate's sign flips, so they do not directly encode the sign interaction.

## Exercise 5.1 — One hidden layer for wine quality

### Code cells 1–3 · Load the same data

These cells import NumPy and Matplotlib, download the CSV if necessary, and make the same seeded split as in 4.1. Using the same 1,898 test wines lets us compare the two RMSE values directly.

### Code cell 4 · Standardize features and initialize parameters

- `X_train.mean(axis=0)` and `std(axis=0)` compute one mean and standard deviation for each column. Subtracting the mean centers the column. Dividing by the standard deviation brings measurements with very different units onto comparable scales. `X_test` uses the same training statistics because those are the only statistics available during training.
- `initialize(hidden_nodes, seed=1234)` sets up one reproducible starting point for each width. `n_inputs` is 11. `W` has shape `(hidden_nodes, 11)`, one row per hidden unit. `b` has one bias per unit. `Wp` has shape `(hidden_nodes, 1)`, with one weight from each hidden unit into the single prediction. `bp` is length one.
- The square-root factors scale the random weights by the size of the layer feeding into them. The small hidden biases break symmetry. The output bias starts near the mean training quality, a sensible constant starting value. The shape printout checks the four arrays before training.

### Code cells 5–6 · ReLU and the forward pass

- `relu(x)` returns the larger of each entry and zero. Positive values pass through. Negative values become zero.
- `z = x @ W.T + b` computes the weighted inputs for all hidden units. For a batch, `x` is `(n, 11)` and `W.T` is `(11, hidden_nodes)`, so `z` is `(n, hidden_nodes)`. A single input vector works with the same expression.
- `h = relu(z)` applies the nonlinearity. `y_pred = h @ Wp + bp` combines the hidden values into one quality prediction per wine. A linear output can predict continuous scores.

### Code cell 7 · Backpropagation and one update

- `z = W @ x + b` is the single-example version of the hidden calculation. `hidden = relu(z)` gives its activations. The gradient function needs both arrays.
- `output_error = 2 * (hidden @ Wp + bp - y)` differentiates the squared error with respect to the predicted score. Adding the output bias changes the score one-for-one, so this is also `dbp`.
- `hidden_error = (Wp @ output_error) * (z > 0)` applies the chain rule. Each hidden unit gets a share of the output error proportional to its outgoing weight. The Boolean mask then applies the ReLU slope: one above zero and zero otherwise.
- `dW = np.outer(hidden_error, x)` gives one derivative for each pair of hidden unit and feature. `db = hidden_error` gives one per hidden bias. `dWp = np.outer(hidden, output_error)` gives a derivative for each hidden-to-output connection. The gradient arrays match the four parameter shapes.
- `update_weights` calculates all four gradients at the current parameters. The four `*_new` assignments subtract `learning_rate * gradient`, one parameter array at a time. The returned arrays become the starting point for the next wine.

### Code cell 8 · Check the gradients

- Choose the first training example and compute its analytical gradients. `np.ndindex(parameter.shape)` visits every element in each parameter array.
- Save the element, evaluate the loss after adding and subtracting `epsilon`, then restore it. `(plus - minus) / (2 * epsilon)` estimates the local derivative from the two losses.
- `np.testing.assert_allclose` checks the numerical and analytical gradients within tolerance. Passing this check helps catch sign, transpose, and broadcasting mistakes before training.

### Code cell 9 · Train and select a checkpoint

- `mse(prediction, target)` averages the squared errors. `train` starts from fresh random parameters and visits every training wine once per epoch in a shuffled order.
- Each call to `update_weights` uses exactly one example. After all updates for the epoch, the forward pass predicts every training wine. `.ravel()` matches the one-dimensional target vector before calculating MSE.
- `losses` stores the epoch-by-epoch MSE. If the current loss is lower than `best_loss`, save the epoch and the parameter tuple. The update function returns new arrays, so later updates leave the saved checkpoint intact.
- `configs` varies width, learning rate, and epoch count. The table reports the lowest training MSE for each run. `min(runs, key=...)` selects the run with the lowest training MSE, as requested by the exercise. The test scores are evaluated in the next cell.

### Code cell 10 · Evaluate the selected network

- Predict on `X_test` and compute MSE, its square root, and Pearson correlation. `mse(y_train.mean(), y_test)` gives the error for a simple model that predicts the same training mean for every wine.
- The first plot uses one training-loss curve per run and marks the selected epoch. The second figure shows the same test predictions as individual points and as a 2D histogram. The histogram helps show where many points overlap. The diagonal gives a reference for perfect agreement in both views.
- The selected run used 80 hidden units and a learning rate of 0.001. The checkpoint at epoch 56 had training MSE 0.4098. Its test RMSE was 0.710, compared with 0.753 for the linear model. The predictions still cluster around middle scores, which you can see from the scatter and histogram.

