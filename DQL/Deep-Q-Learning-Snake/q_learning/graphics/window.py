"""Window Class

The Window Class is responsible for rendering the screen and 
rendering the snake and cherry on the screen.
The dimensions of the screen are 400x400 pixels 
"""

import pygame

pygame.init()

class Window():
    # constructor
    def __init__(self, block_size=20, dims=400):
        # space that each entity occupies in the screen
        self.block_size = block_size
        # width of the screen
        self.width = dims
        # height of the screen
        self.height = dims
        # dimensions of the screen
        self.screen_dimensions = (self.width, self.height)
        # screen
        self.screen = pygame.display.set_mode(self.screen_dimensions)
        # color of the screen
        self.screen_color = (50, 50, 50)

    # renders the screen
    def draw_screen(self):
        rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(self.screen, self.screen_color, rect)

    # renders the snake
    def draw_snake(self, snake):
        # renders head of the snake
        rect = pygame.Rect(snake.head[1] * self.block_size, snake.head[0] * self.block_size, self.block_size, self.block_size)
        pygame.draw.rect(self.screen, snake.head_color, rect)

        # renders the body of the snake
        for piece in snake.body:
            rect = pygame.Rect(piece[1] * self.block_size, piece[0] * self.block_size, self.block_size, self.block_size)
            pygame.draw.rect(self.screen, snake.body_color, rect)

    # renders the cherry
    def draw_cherry(self, cherry):
        rect = pygame.Rect(cherry.position[1] * self.block_size, cherry.position[0] * self.block_size, self.block_size, self.block_size)
        pygame.draw.rect(self.screen, cherry.color, rect)

    def display_score(self, score):
        font = pygame.font.Font("freesansbold.ttf", 16)
        message = font.render(str(score), True, (255, 255, 0))
        self.screen.blit(message, (200, 10))