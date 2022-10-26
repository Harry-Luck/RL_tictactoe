# From https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/dqn_keras.py

"""ReplayBuffer Class

Replay buffer stores past experiences efficiently and samples them
when needed.
"""

import numpy as np

class ReplayBuffer():
    # constructor
    def __init__(self, max_size, input_dims):
        # maximum memory size
        self.mem_size = max_size
        # data in memory
        self.mem_counter = 0
        # memory of past state
        self.state_memory = np.zeros((self.mem_size,input_dims), dtype=np.float32)
        # memory of new state based on action taken
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        # memory of action taken for a state
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        # memory of reward received for the action
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        # memory if the state is a terminal state
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    # stores the scenario in memory
    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_counter += 1

    # samples certain amount of scenarios from memory
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal
