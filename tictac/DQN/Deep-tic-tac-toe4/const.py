import os
from games.TicTacToe import TicTacToe
from games.Connect4 import Connect4
from File_storage import *

# if true, clear the tree between every simulation session. This helps reducing
# the bias, but it might take longer to converge
CLEAR_TREE_ON_ITERATION = False

# number of epochs in the neural network training step
EPOCHS = 5

EPSILON = 0.1

# if true, run 100 games against a random agent
EVALUATE_ON_ITERATION = True

EVALUATION_EPISODES = 100

# game to select
# options:
#       TIC_TAC_TOE
#       CONNECT4
GAME = 'TIC_TAC_TOE'

# the minimum number of visits for a node to be used in the training of the
# neural network
MIN_VISITS = 20

# the number of times we are doing the simulation, network training cycle.
N_ITERATION_MAIN = 10

# the number of roll-outs in the simulation
N_ROLLOUTS = 1000

# name of the file where the weights of the model are saved
WEIGHTS_PATH = "weights"

cwd = os.getcwd()
cwd = cwd + '\\tensorflow_logs'

def init():
    # create game
    global game
    if GAME == 'TIC_TAC_TOE':
        game = TicTacToe()
    elif GAME == 'CONNECT4':
        game = Connect4()
    else:
        print("game ", GAME ,"not recognized")

    # initialize Monte Carlo tree
    global mct
    mct = load_mct(WEIGHTS_PATH, GAME)
