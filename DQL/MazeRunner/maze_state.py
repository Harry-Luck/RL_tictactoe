import numpy as np
from constants import *

class Qmaze(object):
  def __init__(self, maze, rat=(0, 0)):
    self._maze = np.array(maze)         #stores original maze
    rows, cols = self._maze.shape

    #assuming target position to be the last cell
    self.target = (rows-1, cols-1)

    self.free_cells = [(r, c) for r in range(rows) for c in range(cols) if self._maze[r, c]==1.0]
    self.free_cells.remove(self.target)

    if self._maze[self.target] == 0.0:
      raise Exception('Invalid maze : Target position can not be blocked')
    if rat not in self.free_cells:
      raise Exception('Invalid rat postion : Must sit on free cell')
    
    self.reset(rat)

  def reset(self, rat):
    self.maze = np.copy(self._maze)
    self.rat = rat
    self.maze[self.rat] = rat_mark
    self.state = (self.rat[0], self.rat[1], 'start')
    self.min_reward = -0.5*self.maze.size
    self.total_reward = 0
    self.visited = set()

  def update_state(self, action):
    row, col, mode = self.state

    if(self.maze[row, col]>0):
      self.visited.add((row, col))
    
    valid_actions = self.get_valid_actions()

    if not valid_actions:
      mode = 'blocked'
    elif action in valid_actions:
      mode = 'valid'
      if action == LEFT:
        col-=1
      elif action == UP:
        row-=1
      elif action == RIGHT:
        col+=1
      elif action == DOWN:
        row+=1
    else:
      mode = 'invalid'
    
    self.state = (row, col, mode)
  

  def get_valid_actions(self, rat = None):
    if rat is None:
      row, col, _ = self.state
    else:
      row, col = rat
    nrow, ncol = self.maze.shape

    actions = [LEFT, UP, RIGHT, DOWN]
    if row == 0:
      actions.remove(UP)
    elif row == nrow-1:
      actions.remove(DOWN)
    
    if col == 0:
      actions.remove(LEFT)
    elif col == ncol-1:
      actions.remove(RIGHT)
    
    if col>0 and self.maze[row, col-1]==0:
      actions.remove(LEFT)
    if col<ncol-1 and self.maze[row, col+1]==0:
      actions.remove(RIGHT)
    if row>0 and self.maze[row-1, col]==0:
      actions.remove(UP)
    if row<nrow-1 and self.maze[row+1, col]==0:
      actions.remove(DOWN)
    
    return actions

  def get_reward(self):
    row, col, mode = self.state
    if (row, col) == self.target:
      return rewards['win']
    if mode == 'blocked':
      return self.min_reward - 1
    if mode == 'invalid':
      return rewards['wall']
    if mode == 'valid':
      if (row, col) in self.visited:
        return rewards['revisit']
      else:
        return rewards['cost']
    return 0

  def act(self, action):
    self.update_state(action)
    reward = self.get_reward()
    self.total_reward+=reward
    status = self.game_status()
    envstate = self.observe()
    return envstate, reward, status
  
  def game_status(self):
    if self.total_reward<self.min_reward:
      return 'lose'
    row, col, _ = self.state
    if (row, col) == self.target:
      return 'win'
    
    return 'not_over'
  
  def observe(self):
    canvas = np.copy(self.maze)
    nrow, ncol = canvas.shape
    for r in range(nrow):
      for c in range(ncol):
        if canvas[r, c]>0:
          canvas[r, c]=1
    rat_row, rat_col, _ = self.state
    canvas[rat_row, rat_col] = rat_mark
    envstate = canvas.reshape((1, -1))
    return envstate
