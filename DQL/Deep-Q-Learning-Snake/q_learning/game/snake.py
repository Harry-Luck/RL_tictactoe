"""Snake Class

Class that models the snake in the game. The important attributes of the
snake are its head and body. This class contains methods responsible for 
the movement and positioning of the snake. The aim of the agent is to
control this snake.
"""

class Snake():
    # constructor
    def __init__(self):
        # head of the snake
        self.head = []
        # body of the snake
        self.body = []
        # orientation of the snake
        self.orientation = []
        # coordinates to add new piece to the body
        self.last_pos = []
        # color of the snake
        self.head_color = (200, 200, 200)
        self.body_color = (255, 255, 255)

    # moves the snake
    def move(self, action):
        pos = len(self.body) - 1
        self.last_pos = self.body[pos]
        for i in range(pos, 0, -1):
            self.body[i] = self.body[i-1].copy()
        
        # get the new orientation
        self.orientation = self.get_orientation(self.orientation, action)
        self.body[0] = self.head.copy()
        self.head[0] += self.orientation[0]
        self.head[1] += self.orientation[1]

    # gets the new orientation of the snake based on the action takes
    def get_orientation(self, orientation, action):
        if action == 'F':
            return orientation
        
        elif action == 'R':
            if orientation == [0, 1]:
                return [1, 0]
            elif orientation == [1, 0]:
                return [0, -1]
            elif orientation == [0, -1]:
                return [-1, 0]
            elif orientation == [-1, 0]:
                return [0, 1]

        elif action == 'L':
            if orientation == [0, 1]:
                return [-1, 0]
            elif orientation == [1, 0]:
                return [0, 1]
            elif orientation == [0, -1]:
                return [1, 0]
            elif orientation == [-1, 0]:
                return [0, -1]    

    # increases the length of the snake
    def grow(self):
        self.body.append(self.last_pos)

    # checks if the head collided with the body
    def did_hit_body(self):
        for piece in self.body:
            if piece[0] == self.head[0] and piece[1] == self.head[1]:
                return True
        return False

    # checks is the snake is occupying the specfied coordinated
    def is_occupied(self, coordinates):
        x = coordinates[0]
        y = coordinates[1]
        if x == self.head[0] and y == self.head[1]:
            return True
        else:
            for i in range(len(self.body)):
                if x == self.body[i][0] and y == self.body[i][1]:
                    return True
        return False