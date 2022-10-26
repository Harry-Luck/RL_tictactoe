from games.Game import Game
import numpy as np

class Connect4(Game):

    def __init__(self):
        self.restart()
        super().__init__(6,7,7,126)

    def restart(self, player = 0):
        self.board = np.zeros((6,7), dtype="int")
        self.valid_moves = [x for x in range(7)]
        self.terminal = False
        self.player = player

    def is_valid(self, action):
        return action in self.valid_moves

    def get_valid_moves(self):
        return self.valid_moves

    def invert_board(self):
        self.board = np.remainder(3 - self.board, 3)

    def step(self, action):
        self.player = (self.player + 1) % 2
        row_index = 5 - np.argmin(self.board[:,action][::-1])
        self.board[row_index, action] = 1

        if row_index == 0:
            self.valid_moves.remove(action)

        # check the row for a win
        min_col = max(action - 3, 0)
        max_col = min(action + 3, 6)
        count = 0
        for col in range(min_col, max_col + 1):
            if self.board[row_index, col] == 1:
                count += 1
                if count == 4:
                    self.terminal = True
                    return 1
            else:
                count = 0

        # check the column for a win
        min_row = max(row_index - 3, 0)
        max_row = min(row_index + 3, 5)
        count = 0
        for row in range(min_row, max_row+1):
            if self.board[row, action] == 1:
                count += 1
                if count == 4:
                    self.terminal = True
                    return 1
            else:
                count = 0

        # check the diagonal
        min_diag1 = max(min_row - row_index, min_col - action)
        max_diag1 = min(max_row - row_index, max_col - action)
        count = 0
        for d in range(min_diag1, max_diag1+1):
            if self.board[row_index + d, action + d] == 1:
                count += 1
                if count == 4:
                    self.terminal = True
                    return 1
            else:
                count = 0

        # check the other diagonal
        min_diag2 = max(row_index - max_row, min_col - action)
        max_diag2 = min(row_index - min_row, max_col - action)
        count = 0
        for d in range(min_diag2, max_diag2+1):
            if self.board[row_index - d, action +d] == 1:
                count += 1
                if count == 4:
                    self.terminal = True
                    return 1
            else:
                count = 0

        # check if the game is a tie
        if self.valid_moves == []:
            self.terminal = True

        return 0


    def render(self):
        """
        print the board to screen nicely
        """

        for r in range(7):
            print('\n|, end=""')
            for c in range(6):
                if self.board[i,j] == 1:
                    print(' X |', end="")
                elif self.board[r,c] == 0:
                    print('   |', end="")
                else:
                    print(' O |', end="")
        print('\n')
