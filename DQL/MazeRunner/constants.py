rewards = {
    'cost' : -0.04,       #This is the cost for moving to an adjacent cell
    'wall' : -0.75,
    'boundary' : -0.8,
    'revisit' : -0.25,
    'win' : 1,
}

visited_mark = 0.8
rat_mark = 0.5

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

actions_dict = {
    LEFT : 'left',
    UP : 'up',
    RIGHT : 'right',
    DOWN : 'down',
}

num_actions = len(actions_dict)
epsilon = 0.1      #exploration factor