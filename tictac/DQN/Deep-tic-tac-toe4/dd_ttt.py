"""
Self-learning Tic Tac Toe
Made by Lorenzo Mambretti and Hariharan Sezhiyan
"""
import random
import numpy as np
import tensorflow as tf
import time
import datetime

class State:
    board = np.zeros((3,3))
    terminal = False

def is_valid(action, state):
    if state.board[int(np.floor(action / 3))][action % 3] != 0:
        return False
    else:
        return True

def step(state, action):

    # insert
    state_ = State()
    state_.board = np.copy(state.board)
    row_index = int(np.floor(action / 3))
    col_index = action % 3
    state_.board[row_index][col_index] = 1

    # undecided
    terminal = 1

    # to check for 3 in a row horizontal
    for row in range(3):
        for col in range(3):
            if(state_.board[row][col] != 1):
                terminal = 0
        if(terminal == 1):
            state_.terminal = True
            return +1, state_
        else:
            terminal = 1

    # to check for 3 in a row vertical
    for col in range(3):
        for row in range(3):
            if(state_.board[row][col] != 1):
                terminal = 0
        if(terminal == 1):
            state_.terminal = True
            return +1, state_
        else:
            terminal = 1

    # diagonal top-left to bottom-right
    for diag in range(3):
        if(state_.board[diag][diag] != 1):
            terminal = 0
    if(terminal == 1):
        state_.terminal = True
        return +1, state_
    else:
        terminal = 1

    # diagonal bottom-left to top-right
    for diag in range(3):
        if(state_.board[2 - diag][diag] != 1):
            terminal = 0
    if(terminal == 1):
        state_.terminal = True
        return +1, state_
    else:
        terminal = 1

    # checks if board is filled completely
    for row in range(3):
        for col in range(3):
            if(state_.board[row][col] == 0):
                terminal = 0
                break
    if terminal == 1:
        state_.terminal = True

    return 0, state_

def save(W1, W2, B1, B2):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    filename = "weights "+ str(st)+".npz"
    np.savez(filename, W1, W2, B1, B2)
    print("The file has beeen saved successfully")

def load():
    npzfile = np.load("weights.npz")
    W1 = np.reshape(npzfile['arr_0'], (27, 18))
    W2 = np.reshape(npzfile['arr_1'], (18,9))
    b1 = np.reshape(npzfile['arr_2'], (18))
    b2 = np.reshape(npzfile['arr_3'], (9))
    return w1, w2, b1, b2

   
def invert_board(state):
    state_ = State()
    state_.board = np.copy(state.board)
    state_.terminal = state.terminal
    for row in range(3):
        for col in range(3):
            if(state.board[row][col] == 1):
                state_.board[row][col] = 2
            elif(state.board[row][col] == 2):
                state_.board[row][col] = 1
    return state_

def play_game():
    while(True):
        start_nb = input("If you would like to move first, enter 1. Otherwise, enter 2. ")
        start = int(start_nb)
        state = State()
        state.board = np.zeros((3,3))

        while not state.terminal:
            if start == 1:
                action = int(input("Please enter your move: "))
                while(is_valid(action, state) == False):
                    action = int(input("Please enter a correct move: "))
                start = 0
                r, state = step(state, action)
            else:
                state = invert_board(state)
                action = player1.extract_policy(state)
                start = 1
                r, state = step(state, action)
                r = -r
                state = invert_board(state)

            print(state.board)
                
        if r == 0:
            print ("Tie")
        elif r == 1:
            print ("You won")
        else:
            print ("You lost")

def convert_state_representation(state):
    new_board = np.zeros(27)
    for row in range(3):
        for col in range(3):
            if(state[row][col] == 0):
                new_board[9 * row + 3 * col] = 1
            elif(state[row][col] == 1):
                new_board[9 * row + 3 * col + 1] = 1
            else:
                new_board[9 * row + 3 * col + 2] = 1

    return(new_board)

class DDQN(object):

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 27], name='x')
        self.x_ = tf.placeholder(tf.float32, [None, 27], name='x_')

        # for testing against random play
        self.tie_rate_value = 0.0
        self.win_rate_value = 0.0
        self.loss_rate_value = 0.0

        xavier =  tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32)

        # Q learner
        with tf.name_scope('Q-learner') as scope:
            with tf.name_scope('hidden_layer') as scope:
                self.W1 = tf.Variable(xavier([27, 18]))
                self.b1 = tf.Variable(xavier([18]))
                self.h1 = tf.tanh(tf.matmul(self.x, self.W1) + self.b1)
                self.h1_alt = tf.tanh(tf.matmul(self.x_, self.W1) + self.b1)
            with tf.name_scope('output_layer') as scope:
                self.W2 = tf.Variable(xavier([18,9]))
                self.b2 = tf.Variable(xavier([9]))
                self.y = tf.tanh(tf.matmul(self.h1, self.W2) + self.b2)
                self.y_alt = tf.stop_gradient(tf.tanh(tf.matmul(self.h1_alt, self.W2) + self.b2))
                self.action_t = tf.placeholder(tf.int32, [None, 2])
            self.q_learner = tf.gather_nd(self.y, self.action_t)

        # Q target
        with tf.name_scope('Q-target') as scope:
            with tf.name_scope('hidden_layer') as scope:
                self.W1_old = tf.placeholder(tf.float32, [27, 18], name = 'W1_old')
                self.b1_old = tf.placeholder(tf.float32, [18], name = 'b1_old')
                self.h1_old = tf.tanh(tf.matmul(self.x_, self.W1_old) + self.b1_old, name ='h1')
            with tf.name_scope('output_layer') as scope:
                self.W2_old =tf.placeholder(tf.float32, [18, 9], name='W2_old')
                self.b2_old =tf.placeholder(tf.float32, [9], name='b2_old')
                self.y_old = tf.tanh(tf.matmul(self.h1_old, self.W2_old) + self.b2_old, name='y_old')

            self.l_done = tf.placeholder(tf.bool, [None], name='done')
            self.reward = tf.placeholder(tf.float32, [None], name='reward')
            self.gamma = tf.constant(0.99, name='gamma')
            self.qt_best_action = tf.argmax(self.y_alt, axis = 1, name='qt_best_action')
            self.qt_selected_action_onehot = tf.one_hot(indices = self.qt_best_action, depth = 9)
            self.qt= tf.reduce_sum( tf.multiply( self.y_old, self.qt_selected_action_onehot ) , reduction_indices=[1,] )
            self.q_target = tf.where(self.l_done, self.reward, self.reward + (self.gamma * self.qt), name='selected_max_qt')

        self.loss = tf.losses.mean_squared_error(self.q_target, self.q_learner)
        self.tie_rate = tf.placeholder(tf.float32, name='tie_rate')
        self.win_rate = tf.placeholder(tf.float32, name='win_rate')
        self.loss_rate = tf.placeholder(tf.float32, name='loss_rate')
        self.train_step = tf.train.RMSPropOptimizer(0.00020, momentum=0.95, use_locking=False, centered=False, name='RMSProp').minimize(self.loss)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('tie_rate', self.tie_rate)
        tf.summary.scalar('win_rate', self.win_rate)
        tf.summary.scalar('loss_rate', self.loss_rate)
        self.merged = tf.summary.merge_all()
        
        tf.global_variables_initializer().run()

    def update_old_weights(self):
        self.saved_W1 = self.W1.eval()
        self.saved_W2 = self.W2.eval()
        self.saved_b1 = self.b1.eval()
        self.saved_b2 = self.b2.eval()

    def compute_Q_values(self,state):
        # computes associated Q value based on NN function approximator
        q_board = [np.copy(convert_state_representation(state.board))]

        #NN forward propogation
        q_values = sess.run(self.y, {self.x: q_board})
        q_values = np.reshape(q_values, 9)
        return (q_values)

    def extract_policy(self,state):
        policy = None
        q_values = self.compute_Q_values(state)
        for action in range(9):
            if is_valid(action,state):
                if policy == None:
                    policy = action
                    best_q = q_values[action]
                else:
                    new_q = q_values[action]
                    if new_q > best_q:
                        policy = action
                        best_q = new_q
        return policy

    def train(self):
        for _ in range(4):
            # take a random mini_batch
            mini_batch = experience_replay[np.random.choice(experience_replay.shape[0], batch_size), :]

            # select state, state_, action, and reward from the mini batch
            state = np.concatenate(mini_batch[:,0]).reshape((batch_size, -1))
            a = np.transpose(np.append([np.arange(batch_size)],[np.array(mini_batch[:,1])], axis = 0))
            r = mini_batch[:,2]
            state_ = np.concatenate(mini_batch[:,3]).reshape((batch_size, -1))
            done = mini_batch[:,4]
                
            # is the list of all rewards within the mini_batch
            summary, _= sess.run([self.merged, self.train_step], {  self.x: state,
                                                                    self.x_ : state_,
                                                                    self.W1_old : self.saved_W1,
                                                                    self.W2_old : self.saved_W2,
                                                                    self.b1_old : self.saved_b1,
                                                                    self.b2_old : self.saved_b2,
                                                                    self.l_done : done,
                                                                    self.reward : r, 
                                                                    self.action_t : a,
                                                                    self.tie_rate : self.tie_rate_value,
                                                                    self.win_rate : self.win_rate_value,
                                                                    self.loss_rate : self.loss_rate_value})
        train_writer.add_summary(summary, e)

    def random_play_test(self):
        numTests = 100
        numWins = 0
        numLosses = 0
        numTies = 0
        state = State()

        for _ in range(numTests):
            state.board = np.zeros((3,3))
            state.terminal = False
            turn = 1
            while not state.terminal:
                if turn == 1:
                    action = self.extract_policy(state) # agent action
                    r, state = step(state, action)
                    turn = 0
                else:
                    state = invert_board(state)
                    action = np.random.randint(9)
                    while(is_valid(action, state) == False):
                        action = np.random.randint(9)
                    r, state = step(state, action)
                    r = -r
                    state = invert_board(state)
                    turn = 1

            if r == 0:
                numTies += 1
            elif r == 1:
                numWins += 1
            else:
                numLosses += 1

        self.tie_rate_value = numTies
        self.win_rate_value = numWins
        self.loss_rate_value = numLosses

sess = tf.InteractiveSession()
train_writer = tf.summary.FileWriter('tensorflow_logs', sess.graph)


# Global variables
global experience_replay
global batch_size
global e

# Hyperparameters
batch_size = 64
episodes = 100000
epsilon_minimum = 0.1
n0 = 100
start_size = 500
update_target_rate = 50

# Create experience_replay
experience_replay = np.zeros((0,5))

print("All set. Start playing")

# Create players
player1 = DDQN()
#player2 = DDQN() not used yet *** future improvements coming

for e in range(episodes):
    # print("episode ",e)
    state = State()
    if e >= start_size:
        epsilon = max(n0 / (n0 + (e - start_size)), epsilon_minimum)
    else:
        epsilon = 1
    
    if e % 2 == 1:
        # this is player 2's turn
            state = invert_board(state)
            if random.random() < epsilon:
                # take random action
                action_pool = np.random.choice(9,9, replace = False)
                for a in action_pool:
                    if is_valid(a, state):
                        action = a
                        break
            else:
                # take greedy action
                action = player1.extract_policy(state)

            r, state = step(state, action)
            state = invert_board(state)
            r = -r 
    
    while not state.terminal:
        # this section is player 1's turn
        # select epsilon-greedy action
        if random.random() < epsilon:
            # take random action
            action_pool = np.random.choice(9,9, replace = False)
            for a in action_pool:
                if is_valid(a, state):
                    action = a
                    break
        else:
            # take greedy action
            action = player1.extract_policy(state)

        r, state_ = step(state, action)

        if not state_.terminal:
            # this is player 2's turn
            state_ = invert_board(state_)
            if random.random() < epsilon:
                # take random action
                action_pool = np.random.choice(9,9, replace = False)
                for a in action_pool:
                    if is_valid(a, state_):
                        action2 = a
                        break
            else:
                # take greedy action
                action2 = player1.extract_policy(state_) # in the future, it will be player2

            r, state_ = step(state_, action2)
            state_ = invert_board(state_)
            r = -r 

        s = convert_state_representation(np.copy(state.board))
        s_ = convert_state_representation(np.copy(state_.board))
        done = state_.terminal
        D = (s, action, r, s_, done)
        experience_replay = np.append(experience_replay, [D], axis = 0)
        state.board = np.copy(state_.board)
        state.terminal = state_.terminal

    if e == start_size: print("Start Training")
    if e >= start_size:
        if (e % update_target_rate == 0):
            print(e)
            # here save the W1,W2,b1,B2
            player1.update_old_weights()
            player1.random_play_test()

        player1.train()


print("Training completed")
save(player1.W1.eval(), player1.W2.eval(), player1.b1.eval(), player1.b2.eval())
play_game()
