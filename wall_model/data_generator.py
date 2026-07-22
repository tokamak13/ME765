import numpy as np 
import tensorflow as tf
import math

import numpy as np

def wallmodel_utau(Up, yp, nu, kappa=0.41, B=5.2):

    Up = np.abs(Up)

    utau = np.maximum(0.05 * Up, 1e-8)

    for _ in range(20):
        yplus = np.maximum(utau * yp / nu, 1.1)

        f = utau * (np.log(yplus) / kappa + B) - Up
        df = (np.log(yplus) / kappa + B) + 1.0 / kappa

        delta = f / df
        utau -= delta

        if np.max(np.abs(delta)) < 1e-12:
            break

    return utau

def wallmodel_epsilon(Up, yp, nu, kappa=0.41, B=5.2):
    return wallmodel_utau(Up, yp, nu)**3 / (kappa * yp)

def wallmodel_nut(Up, yp, nu, kappa=0.41, B=5.2):
    return kappa * yp * wallmodel_utau(Up, yp, nu)

def generate_data(input_u, input_y, input_nu):
    u, y, nu = np.meshgrid(input_u, input_y, input_nu)

    result_utau = wallmodel_utau(u, y, nu)
    result_epsilon = wallmodel_epsilon(u, y, nu)
    result_nut = wallmodel_nut(u, y, nu)

    #makes arrays linear
    u = u.ravel()
    y = y.ravel()
    nu = nu.ravel()
    result_utau = result_utau.ravel()
    result_epsilon = result_utau.ravel()
    result_nut = result_nut.ravel()

    inputs = np.column_stack((u, y, nu))
    outputs = np.column_stack((result_utau, result_epsilon, result_nut))

    return inputs, outputs

def train_model(input_matrix, output_vector):
    input_normalizer = tf.keras.layers.Normalization(axis=-1)
    input_normalizer.adapt(input_matrix)

    new_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3,)),

        # physical input -> normalized input
        input_normalizer,

        tf.keras.layers.Dense(32, activation="tanh"),
        tf.keras.layers.Dense(64, activation="tanh"),
        tf.keras.layers.Dense(64, activation="tanh"),
        tf.keras.layers.Dense(64, activation="tanh"),
        tf.keras.layers.Dense(32, activation="tanh"),

        # network predicts normalized output
        tf.keras.layers.Dense(1),
    ])

    new_model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = new_model.fit(
        input_matrix,
        output_vector,
        epochs=500,
        batch_size=256,
        validation_split=0.2,
        verbose=1, 
        callbacks=[callback]
    )

    return new_model, history

#train model
input_u = np.linspace(-10, 10, 100)
input_y = np.linspace(0.0001, 0.1, 100)**1.5
input_nu = np.array(0.0000185/1.2) #const for now

training_in, training_out = generate_data(input_u, input_y, input_nu)

utau_model, utau_hist = train_model(training_in, training_out[:,0])
epsilon_model, epsilon_hist = train_model(training_in, training_out[:,1])
nut_model, nut_hist = train_model(training_in, training_out[:,2])

#test model
testing_u = np.linspace(-10, 10, 37)
testing_y = np.linspace(0.0001, 0.1, 100)**1.53
testing_nu = np.array(0.0000185/1.2)

testing_in, testing_out = generate_data(
    testing_u,
    testing_y,
    testing_nu
)

testing_utau = utau_model.predict(testing_in, verbose=0)
testing_epsilon = epsilon_model.predict(testing_in, verbose=0)
testing_nut = nut_model.predict(testing_in, verbose=0)

error_utau = np.sqrt(np.sum((testing_out[:, 0:1] - testing_utau)**2))/testing_out[:, 0:1].size
error_epsilon = np.sqrt(np.sum((testing_out[:, 1:2] - testing_epsilon)**2))/testing_out[:, 1:2].size
error_nut = np.sqrt(np.sum((testing_out[:, 2:3] - testing_nut)**2))/testing_out[:, 2:3].size

error_rel_utau = (
    np.sqrt(np.mean((testing_out[:, 0:1] - testing_utau)**2))
    / np.mean(np.abs(testing_utau))
)

error_rel_epsilon = (
    np.sqrt(np.mean((testing_out[:, 1:2] - testing_epsilon)**2))
    / np.mean(np.abs(testing_epsilon))
)

error_rel_nut = (
    np.sqrt(np.mean((testing_out[:, 2:3] - testing_nut)**2))
    / np.mean(np.abs(testing_nut))
)

print(f"Error: {error_utau:.6e}, {error_epsilon:.6e}, {error_nut:.6e}")
print(f"Rel Error: {error_rel_utau:.6e}, {error_rel_epsilon:.6e}, {error_rel_nut:.6e}")

#save model
utau_model.save("utau_model.keras")
epsilon_model.save("epsilon_model.keras")
nut_model.save("nut_model.keras")


