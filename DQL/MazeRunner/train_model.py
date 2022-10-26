import datetime, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import PReLU
from constants import *


#import maze_state
from maze_state import Qmaze


def play_game(model, qmaze, rat_cell):
  qmaze.reset(rat_cell)
  envstate = qmaze.observe()
  while True:
    prev_envstate = envstate
    q = model.predict(prev_envstate)
    action = np.argmax(q[0])

    envstate, reward, status = qmaze.act(action)
    if status == 'win':
      return True
    elif status =='lose':
      return False

def completion_check(model, qmaze):
    for cell in qmaze.free_cells:
        if not qmaze.get_valid_actions(cell):
            return False
        if not play_game(model, qmaze, cell):
            return False
    return True

class Experience(object):
  def __init__(self, model, max_memory = 100, discount = 0.95):
    self.model = model
    self.max_memory = max_memory
    self.discount = discount
    self.memory = list()
    self.num_actions = model.output_shape[-1]
  
  def remember(self, episode):
    self.memory.append(episode)
    if(len(self.memory)>self.max_memory):
      self.memory.pop(0)

  def predict(self, envstate):
    return self.model.predict(envstate)[0]
  
  def get_data(self, data_size = 10):
    env_size = self.memory[0][0].shape[1]
    mem_size = len(self.memory)
    data_size = min(mem_size, data_size)

    inputs = np.zeros((data_size, env_size))
    targets = np.zeros((data_size, self.num_actions))

    for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
      envstate, action, reward, envstate_next, game_over = self.memory[j]
      inputs[i] = envstate
      targets[i] = self.predict(envstate)
      Q_sa = np.max(self.predict(envstate_next))
      if game_over:
        targets[i, action] = reward
      else:
        targets[i, action] = reward + self.discount * Q_sa
    return inputs, targets

def qtrain(model, maze, **opt):
  global epsilon
  n_epoch = opt.get('n_epoch', 15000)
  max_memory = opt.get('max_memory', 1000)
  data_size = opt.get('data_size', 50)
  weights_file = opt.get('weights_file', "")
  name = opt.get('name', 'model')
  start_time = datetime.datetime.now()
  qmaze = Qmaze(maze)
  
  experience = Experience(model, max_memory=max_memory)
  win_history = []
  n_free_cells = len(qmaze.free_cells)
  hsize = qmaze.maze.size//2   # history window size
  win_rate = 0.0
  imctr = 1

  for epoch in range(n_epoch):
    loss = 0.0
    rat_cell = random.choice(qmaze.free_cells)
    qmaze.reset(rat_cell)
    game_over = False

    # get initial envstate (1d flattened canvas)
    envstate = qmaze.observe()

    n_episodes = 0
    while not game_over:
      valid_actions = qmaze.get_valid_actions()
      if not valid_actions: break
      prev_envstate = envstate
      # Get next action
      if np.random.rand() < epsilon:
          action = random.choice(valid_actions)
      else:
          action = np.argmax(experience.predict(prev_envstate))

      # Apply action, get reward and new envstate
      envstate, reward, game_status = qmaze.act(action)
      if game_status == 'win':
          win_history.append(1)
          game_over = True
      elif game_status == 'lose':
          win_history.append(0)
          game_over = True
      else:
          game_over = False

      # Store episode (experience)
      episode = [prev_envstate, action, reward, envstate, game_over]
      experience.remember(episode)
      n_episodes += 1

      # Train neural network model
      inputs, targets = experience.get_data(data_size=data_size)
      h = model.fit(
        inputs,
        targets,
        epochs=8,
        batch_size=16,
        verbose=0,
      )
      loss = model.evaluate(inputs, targets, verbose=0)
    if len(win_history) > hsize:
      win_rate = sum(win_history[-hsize:]) / hsize
    
    dt = datetime.datetime.now() - start_time
    t = format_time(dt.total_seconds())
    template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
    print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))
    # we simply check if training has exhausted all free cells and if in all
    # cases the agent won
    if win_rate > 0.9 : epsilon = 0.05
    if sum(win_history[-hsize:]) == hsize and completion_check(model, qmaze):
      print("Reached 100%% win rate at epoch: %d" % (epoch,))
      break

def format_time(seconds):
  if seconds < 400:
    s = float(seconds)
    return "%.1f seconds" % (s,)
  elif seconds < 4000:
    m = seconds / 60.0
    return "%.2f minutes" % (m,)
  else:
    h = seconds / 3600.0
    return "%.2f hours" % (h,)

def build_model(maze, lr=0.001):
  model = Sequential()
  model.add(Dense(maze.size, input_shape=(maze.size,)))
  model.add(PReLU())
  model.add(Dense(maze.size))
  model.add(PReLU())
  model.add(Dense(num_actions))
  model.compile(optimizer='adam', loss='mse')
  return model