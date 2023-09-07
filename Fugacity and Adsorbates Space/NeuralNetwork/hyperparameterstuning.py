#!/usr/bin/env python3

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import os

# Load training dataset
train_df = pd.read_csv("GP_training.csv")
train_df = train_df.dropna()  # Drop rows with NaN values
train_x = np.array(train_df.loc[:, "V2":"sig_eff"])
train_y = np.array(train_df.loc[:, "loading"])

# Load validation dataset
valid_df = pd.read_csv("validation.csv")
valid_df = valid_df.dropna()  # Drop rows with NaN values
valid_x = np.array(valid_df.loc[:, "V2":"sig_eff"])
valid_y = np.array(valid_df.loc[:, "loading"])

# Scale inputs/predictors
scaler = MinMaxScaler(feature_range=(0, 1))
train_x = scaler.fit_transform(train_x)
valid_x = scaler.transform(valid_x)

# Define a function to create the model
def create_model(learning_rate=0.0001, epochs=500, batch_size=50):
    model = Sequential()
    model.add(Dense(64, input_dim=31, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(128, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(64, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, kernel_initializer='he_normal'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mean_squared_error', 'mean_absolute_percentage_error'])
    return model

# Define hyperparameters to search
param_grid = {
    'learning_rate': [0.001, 0.0001, 0.00001],
    'epochs': [100, 200, 500],
    'batch_size': [32, 64, 128]
}

# Create the model
model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=create_model, verbose=2)

# Perform GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, verbose=2)
grid_result = grid.fit(train_x, train_y)

# Create a directory to save the models
model_directory = 'saved_models'
os.makedirs(model_directory, exist_ok=True)

# Save all models from every combination
for i, params in enumerate(grid_result.cv_results_['params']):
    model_name = f'model_lr_{params["learning_rate"]}_epochs_{params["epochs"]}_batch_{params["batch_size"]}.h5'
    model_path = os.path.join(model_directory, model_name)
    model = create_model(learning_rate=params['learning_rate'], epochs=params['epochs'], batch_size=params['batch_size'])

    # Train the model
    history = model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=params['epochs'], batch_size=params['batch_size'], verbose=2)

    # Save the model
    model.save(model_path)

    # Plot the learning curve
    fig = plt.figure(dpi=150)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title(f'Model {i + 1} Learning Curve\nEpochs: {params["epochs"]}, Batch Size: {params["batch_size"]}, Learning Rate: {params["learning_rate"]}')
    fig.savefig(f'model_{i + 1}_learning_curve.png')
    plt.close(fig)

    # Close the model session
    tf.keras.backend.clear_session()

# Find the best model based on the lowest validation loss
best_val_loss = float('inf')
best_model_path = None

for root, _, files in os.walk(model_directory):
    for file in files:
        if file.endswith(".h5"):
            model = tf.keras.models.load_model(os.path.join(root, file))
            val_loss = model.evaluate(valid_x, valid_y)[0]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(root, file)

# Load the best model
best_model = tf.keras.models.load_model(best_model_path)

# Save the overall best model
best_model.save('adsorbate_prediction.h5')

# Save learning curve of the best model
best_params = grid_result.cv_results_['params'][np.argmin(grid_result.cv_results_['mean_test_score'])]
history = best_model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=2)

fig = plt.figure(dpi=150)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title(f'Best Model Learning Curve\nEpochs: {best_params["epochs"]}, Batch Size: {best_params["batch_size"]}, Learning Rate: {best_params["learning_rate"]}')
fig.savefig('best_model_learning_curve.png')
plt.close(fig)
