"""
Self-learning Tic Tac Toe
Made by Lorenzo Mambretti

Last Update: Dec 8 2020
"""
import random
import numpy as np
import progressbar
from File_storage import *
from nn import train_model, get_model, save_model
import const
import math
import tensorflow as tf
import argparse

class Node:
    global nnet
    def __init__(self, board):
        self.N = 0
        self.V = 0
        self.Child_nodes = {}
        self.board = board

    def update(self,r):
        self.V += r
        self.N += 1

    def Q(self):
        c_puct = 0.2    #hyperparameter
        P = nnet.value(self.board)

        Q = (self.V + P)/(self.N + 1)
        return Q

def check_new_node(game, current_node, a):
    """
    this function check if it is the first time the player visited the current node
    if it is the first time, create all the child nodes
    and append them in the Monte Carlo Tree (const.mct)
    """
    if a not in current_node.Child_nodes.keys():
        # generate child nodes
        next_ix = len(const.mct)
        current_node.Child_nodes[a] = next_ix
        const.mct[next_ix] = Node(game.board2array())


def random_move(game, current_node):
    #check if it is a new node
    a = random.choice(game.get_valid_moves())
    return a


def choose_move(game, current_node):

    pred = nnet.policy(current_node.board).numpy()[0]

    # else find the best of the nodes
    for a in range(game.action_space):
        if a in current_node.Child_nodes.keys():
            c = current_node.Child_nodes[a]
            pred[a] = const.mct[c].Q()
            continue

        if not game.is_valid(a):
            pred[a] = -2

    return np.argmax(pred)

def evaluation(game, episodes):
    print("evaluation() started")

    defeats = 0
    victories = 0
    for e in range(episodes):

        game.restart(player = e % 2)
        current_node = const.mct[0]

        while not game.terminal:
            # agent to evaluate
            if game.player == 0:
                a = choose_move(game, current_node)
                r = game.step(a)
            else:
                # random agent
                a = random_move(game, current_node)
                game.invert_board()
                r = -game.step(a)
                game.invert_board()

            check_new_node(game, current_node, a)
            current_node = const.mct[current_node.Child_nodes[a]]

        if r == 1:
            victories += 1
        elif r == -1:
            defeats += 1

    return victories, defeats


def simulation(game, episodes):
    print("simulation() started")
    node_list = [[]]

    # progressbar
    bar = progressbar.ProgressBar(maxval=episodes, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for e in range(episodes):

        if (e + 1) % (episodes/100) == 0:
            bar.update(e)

        game.restart(player = e % 2)             # empty board
        node_list.clear()
        current_node = const.mct[0]   # root of the tree is current node
        node_list.append([0,game.player])

        # while state not terminal
        while game.terminal == False:
            # choose move
            if game.player == 0:
                #if player 1 not random
                if random.random() < const.EPSILON:
                    a = random_move(game, current_node)
                else:
                    a = choose_move(game, current_node)
                r = game.step(a)
            else:
                #if player 2 epsilon-greedy
                if random.random() < const.EPSILON:
                    a = random_move(game, current_node)
                else:
                    a = choose_move(game, current_node)
                game.invert_board()
                r = - game.step(a)
                game.invert_board()

            check_new_node(game, current_node, a)
            current_ix = current_node.Child_nodes[a]
            current_node = const.mct[current_ix]
            node_list.append([current_ix,game.player])
            #save state in node list

        #update all nodes
        for node in node_list:
            if node[1] == 0:
                const.mct[node[0]].update(-r)
            else:
                const.mct[node[0]].update(r)

    bar.finish()

def play():
    import gui

def extract_data(game, model, min_visits = 10):
    print("extract_data(): start")

    n_nodes = len(const.mct)
    data = np.empty((n_nodes, game.obs_space))
    policy_labels = np.empty((n_nodes, game.action_space))
    value_labels = np.empty((n_nodes, 1))

    i = 0
    for node in const.mct.values():

        # don't count if does't reach the min required
        if node.N < min_visits:
            continue

        data[i] = node.board

        value_labels[i] = np.array(node.V / node.N)

        pred = model.policy(node.board).numpy()[0]

        for a in range(game.action_space):
            if a in node.Child_nodes.keys():
                c = node.Child_nodes[a]
                pred[a] = const.mct[c].Q()

            if not game.is_valid(a):
                pred[a] = -1

        policy_labels[i] = pred

        i += 1

    data = data[:i,:]
    policy_labels = policy_labels[:i, :]
    value_labels = value_labels[:i, :]
    print("extract_data(): end")

    return data, policy_labels, value_labels

def train():
    # create neural network
    global nnet
    nnet = get_model()

    # SIMULATION: playing and updating Monte Carlo Tree
    if not bool(const.mct):
        # start with a simulation
        const.mct[0] = Node(const.game.board2array())
        simulation(const.game, const.N_ROLLOUTS)

    for _ in range(const.N_ITERATION_MAIN):

        data, p_labels, v_labels = extract_data(const.game, nnet, const.MIN_VISITS)

        # train the network
        train_model(data, p_labels, v_labels, nnet, const.EPOCHS)

        # clear the monte carlo tree
        if (const.CLEAR_TREE_ON_ITERATION):
            const.game.restart()
            const.mcd = {
                0: Node(const.game.board2array())
            }

        if (const.EVALUATE_ON_ITERATION):
            w, l = evaluation(const.game, const.EVALUATION_EPISODES)
            t = const.EVALUATION_EPISODES - (w + l)
            print("won: ", w, " ties: ", t, " lost: ",l)

        simulation(const.game, const.N_ROLLOUTS)
        # save the Monte Carlo Tree

    # save model
    save_model(nnet)

    # save latest monte carlo tree
    save_mct(const.mct, const.WEIGHTS_PATH, const.GAME)


if __name__ == "__main__":
    const.init()

    parser = argparse.ArgumentParser(description='Train or play.')
    parser.add_argument('--play', dest='accumulate', action='store_const',
                       const=play, default=train,
                       help='play a const.game (default: train)')

    args = parser.parse_args()
    print(args.accumulate())
