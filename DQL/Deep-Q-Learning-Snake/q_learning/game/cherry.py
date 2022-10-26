"""Cherry Class

Class that models the food in the game. Using a class for such a simple
model may be an overkill. But it improves the readability of the code
and it made the coding of the application less confusing.
"""

class Cherry():
    # constructor
    def __init__(self, position):
        # location of the cherry in the grid world
        self.position = position
        # color of the cherry
        self.color = (255, 0, 0)