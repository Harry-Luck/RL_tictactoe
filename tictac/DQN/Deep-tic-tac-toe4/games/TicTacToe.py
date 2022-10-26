from games.Game import Game
import numpy as np

class TicTacToe(Game):

    def __init__(self):
        self.restart()
        super().__init__(3,3,9,27)

    def restart(self, player = 0):
        self.board = np.zeros((3,3),dtype="int")
        self.valid_moves = [x for x in range(9)]
        self.terminal = False
        self.player = player

    def is_valid(self, action):
        if self.board[int(np.floor(action / 3))][action % 3] != 0:
            return False
        else:
            return True

    def get_valid_moves(self):
        return self.valid_moves

    def invert_board(self):
        self.board = np.remainder(3 - self.board, 3)

    def step(self,action):
        """
        PARAMS: a valid action (int 0 to 8)
        RETURN: reward (-1,0,1)

        self.board    is updated in the process
        self.terminal is updated in the process
        """
        self.player = (self.player + 1) % 2
        self.valid_moves.remove(action)

        # insert
        row_index = int(np.floor(action / 3))
        col_index = action % 3
        self.board[row_index][col_index] = 1

        # undecided
        terminal = 1

        # to check for 3 in a row horizontal
        for col in range(3):
            if(self.board[row_index][col] != 1):
                terminal = 0
                break
        if(terminal == 1):
            self.terminal = True
            return +1
        else:
            terminal = 1

        # to check for 3 in a row vertical
        for row in range(3):
            if(self.board[row][col_index] != 1):
                terminal = 0
        if(terminal == 1):
           self.terminal = True
           return +1
        else:
           terminal = 1

        # diagonal top-left to bottom-right
        for diag in range(3):
            if(self.board[diag][diag] != 1):
                terminal = 0
        if(terminal == 1):
            self.terminal = True
            return +1
        else:
            terminal = 1

        # diagonal bottom-left to top-right
        for diag in range(3):
            if(self.board[2 - diag][diag] != 1):
                terminal = 0
                break
        if(terminal == 1):
            self.terminal = True
            return +1

        # checks if board is filled completely
        if 0 not in self.board:
            self.terminal = True

        return 0

    def render(self):
        """
        print to screen the full board nicely
        """

        for i in range(3):
            print('\n|', end="")
            for j in range(3):
                if self.board[i][j] == 1:
                    print(' X |' , end="")
                elif self.board[i][j] == 0:
                    print('   |' , end="")
                else:
                    print(' O |' , end="")
        print('\n')
