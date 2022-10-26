"""Class Agent

The Agent is the main controller of the game.
It contains a function approximator (neural newtork) that it
uses to predict the Q-values of the state and chooses an appropriate
action based on its policy. The epsilon value indicates the degree of
exploration vs exploitation. Over time the degree of exploration reduces
and the degree of exploitation increases. This is done using a decay value.
"""

from agent.memory import ReplayBuffer
from agent.network import create_model
from tensorflow.keras.models import load_model
import numpy as np


class Agent():
    # constructor
    def __init__(self, lr, gamma, action_size, epsilon, batch_size, input_dims, epsilon_decay=0.995, min_epsilon=0.001, mem_size = 1000000, filename='q_learning_model.h5'):
        # Possible actions
        self.action_space = [i for i in range(action_size)]
        # Epsilon value
        self.epsilon = epsilon
        # Discount Factor
        self.gamma = gamma
        # Batch size to train on
        self.batch_size = batch_size
        # Epsilon decay
        self.epsilon_decay = epsilon_decay
        # Lowest possible epsilon value
        self.min_epsilon= min_epsilon
        # Filename to store the neural network
        self.filename = filename
        # Memory to store past experiences
        self.memory = ReplayBuffer(mem_size, input_dims)
        # Function Approximator : neural network
        self.model = create_model(lr, action_size, input_dims, 256, 256)

    # stores the recent experience in memory
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    # choses and action based on the agent's policy
    def choose_action(self, observation):
        # explore
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        # exploit
        else:
            state = np.array([observation])
            actions = self.model.predict(state)
            action = np.argmax(actions)   # Greedy action

        return action

    # updates the weights of the neural newtork to approximate more accurately

    ################################################################
    # from https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/dqn_keras.py
    ################################################################
    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        
        # sample some experiences from memory
        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)
        
        q_value = self.model.predict(states)  # get current q-values
        q_next = self.model.predict(new_states) # get q value of the next state

        q_target = np.copy(q_value)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # update based on Bellman equations
        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis = 1) * (1 - dones)

        # updates weights
        self.model.train_on_batch(states, q_target)

        # Reducing degree of exploration and increasing degree of exploitation
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.min_epsilon
    
    # saves the model to the file
    def save_model(self):
        self.model.save(self.filename)
    
    # loads the model from the file for further training
    def load_model(self):
        self.model = load_model(self.filename)
