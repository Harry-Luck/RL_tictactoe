"""Helper Functions File

This module contains helper functions that help in managing
the game and environment checks.
"""

# TODO: Add did_hit_boundary functionality

import random

# checks if the snake hit the wall
def did_hit_boundary(snake, lower_end, upper_end):
    return snake.head[0] < lower_end or snake.head[0] >= upper_end or snake.head[1] < lower_end or snake.head[1] >= upper_end

# generates a new location for the cherry if the snake has not occupied the location
def get_cherry_location(snake):
    while True:
        new_location = generate_cherry()
        if not snake.is_occupied(new_location):
            return new_location

# checks if the snake ate the cherry
def did_eat_cherry(snake, cherry):
    return snake.head[0] == cherry.position[0] and snake.head[1] == cherry.position[1]

# gets a candidate location for the cherry
def generate_cherry():
    list_x = [x for x in range(0, 20, 1)]
    list_y = [y for y in range(0, 20, 1)]
    x = random.choice(list_x)
    y = random.choice(list_y)
    return [x, y]