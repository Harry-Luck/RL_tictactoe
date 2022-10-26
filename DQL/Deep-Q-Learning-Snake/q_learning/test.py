import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import pygame
from game.env import Environment
from graphics.window import Window


pygame.init()

MODEL_FILE = 'q_learning_model.h5'

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    env = Environment()
    agent = load_model(MODEL_FILE)
    print(agent.summary())

    window = Window()
    score = 0
    observation = env.reset()

    pygame.display.set_caption('Deep Q-Learning Snake')
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        window.draw_screen()
        window.draw_snake(env.snake)
        window.draw_cherry(env.cherry)
        window.display_score(score)
        pygame.time.delay(50)
        pygame.display.update()
        state = np.array([observation])
        actions = agent.predict(state)
        action = np.argmax(actions)
        new_observation, reward, done = env.step(action)
        if reward == 10:
            score += 1
        observation = new_observation
