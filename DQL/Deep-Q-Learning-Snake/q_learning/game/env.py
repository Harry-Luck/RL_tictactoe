"""Class that models the environment

The Environment Class is the environment for the Snake Game.
Contains essential environment functions (reset, get_state, step and is_done)
that the agent can use to play and learn. The environment is a 20x20 grid
where the snake is allowed to move. The aim of the game is to collect cherries
and avoid obstacles such as walls and the snake's body itself. The Environment
contains 15 state variables which represent elements in the state list.
The action space is 3 as the snake can only take 3 actions.
The snake can either turn left or right or head in the same direction.
"""


from game.snake import Snake
from game.cherry import Cherry
import game.game_manager as gm
import numpy as np


class Environment():
    # constructor
    def __init__(self):
        # size of the grid world
        self.size = 20
        # state size
        self.state_size = 15
        # state
        self.state = np.zeros(shape=self.state_size)
        # Snake
        self.snake = Snake()
        self.snake.head = [3, 3]
        self.snake.body = [[3, 2], [3, 1]]
        self.snake.orientation = [0, 1]
        self.snake.last_pos = [3, 1]
        # Cherry
        self.cherry = Cherry(gm.get_cherry_location(self.snake))
        # possible actions
        self.actions = 'L', 'R', 'F'  # Left, Right, Forward
        # action space
        self.action_space = len(self.actions)
        # moves for the snake to eat cherry (prevents the snake from getting stuck in a loop)
        self.moves = self.size * self.size
        # possible rewards for the agent
        self.rewards = {
            "dead" : -100,
            "away" : -1,
            "toward" : 0.5,
            "hit" : 10,
            "stall" : -200
        }

    # Updates the state based on the action taken
    def step(self, action):
        done = False
        reward = 0
        # gets old distance from cherry
        dist = abs(self.snake.head[0] - self.cherry.position[0]) + abs(self.snake.head[1] - self.cherry.position[1]) 
        
        # sanity check for an invalid action
        if not 0 <= action < self.action_space:
            return self.state, reward, done

        # gets the direction to turn to
        direction = self.actions[action]
        self.snake.move(direction) # moves the snake to that direction

        # checks if the snake lost
        if self.is_done():
            reward = self.rewards["dead"]
            done = True
            return self.state, reward, done

        # gets the new distance from cherry
        new_dist = abs(self.snake.head[0] - self.cherry.position[0]) + abs(self.snake.head[1] - self.cherry.position[1]) 

        if new_dist > dist:
            reward = self.rewards["away"]
        else:
            reward = self.rewards["toward"]

        # checks if the snake ate the cherry
        if gm.did_eat_cherry(self.snake, self.cherry):
            self.moves = self.size * self.size  #resets the moves
            self.snake.body.append(self.snake.last_pos)
            reward = self.rewards["hit"]
            self.cherry.position = gm.get_cherry_location(self.snake)

        self.moves -= 1
        # checks if the snake has exhausted its moves
        if self.moves == 0:
            done = True
            reward = self.rewards["stall"]

        self.state = self.get_state() # updates the state

        return self.state, reward, done

    # resets the environment
    def reset(self):
        self.snake.head = [3, 3]
        self.snake.body = [[3, 2], [3, 1]]
        self.snake.orientation = [0, 1]
        self.snake.last_pos = [3, 1]
        self.cherry = Cherry(gm.get_cherry_location(self.snake))
        self.state = self.get_state()
        return self.state

    # returns the state
    def get_state(self):
        state = np.zeros(self.state_size)
        state[0] = self.cherry.position[0]/self.size
        state[1] = self.cherry.position[1]/self.size
        state[2] = self.snake.orientation[0]
        state[3] = self.snake.orientation[1]
        state[4], state[5], state[6], state[7] = self.get_relative_direction()
        state[8], state[9] = self.snake.head[0]/self.size, self.snake.head[1]/self.size
        state[10] = self.moves / (self.size * self.size)
        state[11], state[12], state[13], state[14] = self.get_obstacles()
        return state

    # checks if the snake died
    def is_done(self):
        return self.snake.did_hit_body() or gm.did_hit_boundary(self.snake, lower_end=0, upper_end=self.size)

    # gets the relative position of the cherry from the snake    
    def get_relative_direction(self):
        front, back, left, right = 0, 0, 0, 0

        # Rightwards orientation
        if self.snake.orientation == [0, 1]:
            if self.cherry.position[0] > self.snake.head[0]:
                right = 1
            elif self.cherry.position[0] < self.snake.head[0]:
                left = 1
            
            if self.cherry.position[1] > self.snake.head[1]:
                front = 1
            
            elif self.cherry.position[1] < self.snake.head[1]:
                back = 1

        # Downward orientation
        elif self.snake.orientation == [1, 0]:
            if self.cherry.position[0] > self.snake.head[0]:
                front = 1
            elif self.cherry.position[0] < self.snake.head[0]:
                back = 1
            
            if self.cherry.position[1] > self.snake.head[1]:
                left = 1
            
            elif self.cherry.position[1] < self.snake.head[1]:
                right = 1

        # Leftwards orientation
        elif self.snake.orientation == [0, -1]:
            if self.cherry.position[0] > self.snake.head[0]:
                left = 1
            elif self.cherry.position[0] < self.snake.head[0]:
                right = 1
            
            if self.cherry.position[1] > self.snake.head[1]:
                back = 1
            
            elif self.cherry.position[1] < self.snake.head[1]:
                front = 1

        # Upward orientation
        elif self.snake.orientation == [-1, 0]:
            if self.cherry.position[0] > self.snake.head[0]:
                back = 1
            elif self.cherry.position[0] < self.snake.head[0]:
                front = 1
            
            if self.cherry.position[1] > self.snake.head[1]:
                right = 1
            
            elif self.cherry.position[1] < self.snake.head[1]:
                left = 1

        return front, back, left, right

    # returns the vision for the snake on all 4 directions
    # TODO increase the vision for the snake so that it doesn't hit itself
    def get_obstacles(self):
        up, down, left, right = 0, 0, 0, 0

        x = self.snake.head[0]
        y = self.snake.head[1]

        a = self.snake.head[0] + 1
        b = self.snake.head[0] - 1

        c = self.snake.head[1] + 1
        d = self.snake.head[1] - 1

        if a >= self.size:
            down = 1
        if b < 0:
            up = 1
        
        if c >= self.size:
            right = 1
        if d < 0:
            left = 1

        if [x, c] in self.snake.body:
            right = 1
        if [x, d] in self.snake.body:
            left = 1

        if [a, y] in self.snake.body:
            down = 1
        if [b, y] in self.snake.body:
            up = 1

        return up, down, left, right