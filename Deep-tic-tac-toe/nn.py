"""
Neural network
Made by Lorenzo Mambretti
"""

import tensorflow as tf
from models.ConnectNN import ConnectNN
from models.TicTacToeNN import TicTacToeNN
print("tensorflow: ",tf.version.VERSION)
import numpy as np
import os
import random
import const

def get_model():
    if const.GAME == 'TIC_TAC_TOE':
        nnet = TicTacToeNN()
    elif const.GAME == 'CONNECT4':
        nnet = ConnectNN()

    filename = os.path.join(const.WEIGHTS_PATH, "checkpoint_" + const.GAME)
    try:
        nnet.load_weights(filename)
        print("get_model(): Load neural network weights")
    except:
        print("get_model():", filename, " not found")

    return nnet

def save_model(model):
    filename = os.path.join(const.WEIGHTS_PATH, "checkpoint_" + const.GAME)
    model.save_weights(filename)

"""
define here the training process and all necessary accessories"""
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

train_policy_loss = tf.keras.metrics.Mean(name='train_policy_loss')
train_value_loss = tf.keras.metrics.Mean(name='train_value_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

@tf.function
def train_policy_step(model, data, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model.policy(data)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_policy_loss(loss)

@tf.function
def train_value_step(model, data, labels):
    with tf.GradientTape() as tape:
        predictions = model.value(data)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_value_loss(loss)

@tf.function
def test_step(model, data, labels):
    predictions = model.policy(data)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)

def train_model(data, p_labels, v_labels, model, epochs, batch_size = 64):

    print("train_model() started")

    # create datasets
    p_train_dataset = tf.data.Dataset.from_tensor_slices((data, p_labels))
    p_train_dataset = p_train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    v_train_dataset = tf.data.Dataset.from_tensor_slices((data, p_labels))
    v_train_dataset = v_train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    for epoch in range(epochs):

        train_policy_loss.reset_states()
        train_value_loss.reset_states()
        test_loss.reset_states()

        for step, (x_batch_train, y_batch_train) in enumerate(p_train_dataset):
            train_policy_step(model, x_batch_train, y_batch_train)

        for step, (x_batch_train, y_batch_train) in enumerate(v_train_dataset):
            train_value_step(model, x_batch_train, y_batch_train)

        print(
            f'Epoch {epoch + 1}, '
            f'Policy Loss: {train_policy_loss.result()}, '
            f'Value Loss: {train_value_loss.result()}, '
          )
