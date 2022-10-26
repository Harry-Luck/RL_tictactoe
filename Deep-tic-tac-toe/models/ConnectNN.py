import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras import regularizers

class ConnectNN(Model):

    def __init__(self):
        super(ConnectNN, self).__init__()
        self.d1 = Dense(100, activation='relu')
        self.d2 = Dense(100, activation='relu')

        self.p1 = Dense(30, activation='relu',
            kernel_regularizer=regularizers.l2(0.0001))
        self.policy_head = Dense(7, activation='tanh')

        self.v1 = Dense(10, activation='relu')
        self.value_head = Dense(1, activation='tanh')

    def body(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return x

    def policy(self, x):
        x = self.body(x)
        x = self.p1(x)
        return self.policy_head(x)

    def value(self, x):
        x = self.body(x)
        x = self.v1(x)
        return self.value_head(x)
