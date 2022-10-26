"""
Self-learning Tic Tac Toe
Made by Lorenzo Mambretti and Hariharan Sezhiyan

"""

import random
import numpy as np
import tensorflow as tf

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
    np.savez("weights.npz", W1, W2, B1, B2)
    print("file weights.txt has beeen updated successfully")

def load():
    npzfile = np.load("weights.npz")
    W1 = np.reshape(npzfile['arr_0'], (27, 18))
    W2 = np.reshape(npzfile['arr_1'], (18,9))
    b1 = np.reshape(npzfile['arr_2'], (18))
    b2 = np.reshape(npzfile['arr_3'], (9))
    return w1, w2, b1, b2

def extract_policy(state):
    policy = None
    q_values = compute_Q_values(state)
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
                action = extract_policy(state)
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

def compute_Q_values(state):
    # computes associated Q value based on NN function approximator
    q_board = np.zeros((1,27))
    q_board = [np.copy(convert_state_representation(state.board))]

    #NN forward propogation
    q_values = sess.run(y, feed_dict = {x: q_board})
    q_values = np.reshape(q_values, 9)
    return (q_values)

def train(experience_replay, saved_W1, saved_W2, saved_b1, saved_b2):
    # can modify batch size here
    batch_size = 32

    # take a random mini_batch
    mini_batch = experience_replay[np.random.choice(experience_replay.shape[0], batch_size), :]

    # select state, state_, action, and reward from the mini batch
    state = np.concatenate(mini_batch[:,0]).reshape((batch_size, -1))
    act = np.array(mini_batch[:,1])
    act = np.append([np.arange(batch_size)],[act], axis = 0)
    act = np.transpose(act)
    r = mini_batch[:,2]
    state_ = np.concatenate(mini_batch[:,3]).reshape((batch_size, -1))
    done = mini_batch[:,4]
    
    # is the list of all rewards within the mini_batch
        
    summary, _= sess.run([merged, train_step], feed_dict={  x: state,
                                                            x_old : state_,
                                                            W1_old : saved_W1,
                                                            W2_old : saved_W2,
                                                            b1_old : saved_b1,
                                                            b2_old : saved_b2,
                                                            l_done : done,
                                                            reward : r, 
                                                            action_t : act
                                                            })
    train_writer.add_summary(summary)

# Q learner neural network
with tf.name_scope('Q-learner') as scope:
    x = tf.placeholder(tf.float32, [None, 27], name='x')
    with tf.name_scope('hidden_layer') as scope:
        W1 = tf.get_variable("W1", shape=[27, 18],
           initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", shape=[18],
           initializer=tf.contrib.layers.xavier_initializer())
        h1 = tf.tanh(tf.matmul(x, W1) + b1)
    with tf.name_scope('output_layer') as scope:
        W2 = tf.get_variable("W2", shape=[18, 9],
           initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2", shape=[9],
           initializer=tf.contrib.layers.xavier_initializer())
        y = tf.tanh(tf.matmul(h1, W2) + b2)
        action_t = tf.placeholder(tf.int32, [None, 2])

    q_learner = tf.gather_nd(y, action_t)

# Q target neural network
with tf.name_scope('Q-target') as scope:
    x_old = tf.placeholder(tf.float32, [None, 27], name='x_old')
    with tf.name_scope('hidden_layer') as scope:
        W1_old = tf.placeholder(tf.float32, [27, 18], name='W1_old')
        b1_old = tf.placeholder(tf.float32, [18], name='b1_old')
        h1_old = tf.tanh(tf.matmul(x_old, W1_old) + b1_old, name='h1')
    with tf.name_scope('output_layer') as scope:
        W2_old =tf.placeholder(tf.float32, [18, 9], name='W2_old')
        b2_old =tf.placeholder(tf.float32, [9], name='b2_old')
        y_old = tf.tanh(tf.matmul(h1_old, W2_old) + b2_old, name='y_old')

    l_done = tf.placeholder(tf.bool, [None])
    reward = tf.placeholder(tf.float32, [None])
    gamma = tf.constant(0.99, name='gamma')
    qt = tf.reduce_max(y_old, axis = 1, name='maximum_qt')
    q_target = tf.where(l_done, reward, reward + (gamma * qt), name='selected_max_qt')

with tf.name_scope('loss') as scope:
    loss = tf.losses.mean_squared_error(q_target, q_learner)
    
#train_step = tf.train.GradientDescentOptimizer(0.03).minimize(loss)
train_step = tf.train.RMSPropOptimizer(0.00025, momentum=0.95, use_locking=False, centered=False, name='RMSProp').minimize(loss)

tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()

sess = tf.InteractiveSession()
train_writer = tf.summary.FileWriter('tensorflow_logs', sess.graph)
tf.global_variables_initializer().run()

episodes = 100000
n0 = 100.0
start_size = 500
experience_replay = np.zeros((0,5))

print("All set. Start epoch")

for e in range(episodes):
    # print("episode ",e)
    state = State()
    if e >= start_size:
        epsilon = max(n0 / (n0 + (e- start_size)), 0.1)
    else: epsilon = 1
    
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
                action = extract_policy(state)

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
            action = extract_policy(state)

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
                action2 = extract_policy(state_)

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
        if((e % 50) == 0):
            print("Episode:",e)
            # here save the W1,W2,b1,B2
            saved_W1 = W1.eval()
            saved_W2 = W2.eval()
            saved_b1 = b1.eval()
            saved_b2 = b2.eval()
        train(experience_replay, saved_W1, saved_W2, saved_b1, saved_b2)
print("Training completed")
play_game()
