"""Neural Network

create_model function returns the neural network that will be used
to approximate the Q-values. Model is built using tensorflow and keras
"""

from tensorflow import keras
from tensorflow.keras.optimizers import Adam

def create_model(lr, n_actions, input_dims, layer_1_dims, layer_2_dims):
    model = keras.Sequential([
        keras.layers.Dense(layer_1_dims, activation='relu'),
        keras.layers.Dense(layer_2_dims, activation='relu'),
        keras.layers.Dense(n_actions, activation=None)
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mean_squared_error'
    )

    print(model.summary)
    return model