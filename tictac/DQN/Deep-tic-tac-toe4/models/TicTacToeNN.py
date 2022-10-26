import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class TicTacToeNN(Model):

    def __init__(self):
        super(TicTacToeNN, self).__init__()
        self.d1 = Dense(20, activation='relu')
        self.d2 = Dense(20, activation='relu')

        self.policy_head = Dense(9, activation='tanh')
        self.value_head = Dense(1, activation='tanh')

    def body(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return x

    def policy(self, x):
        x = self.body(x)
        return self.policy_head(x)

    def value(self, x):
        x = self.body(x)
        return self.value_head(x)
