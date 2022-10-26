import pygame
import numpy as np
import tensorflow as tf
import pickle
from graphics.window import Window
from agent.agent import Agent
from game.env import Environment

pygame.init()


LEARNING_RATE = 0.00025
EPSILON = 1
NUMBER_OF_GAMES = 500
EPSILON_END = 0.001
GAMMA = 0.95
EPSILON_DECAY = 0.9995
FILE_NAME = "q_learning_model.h5"
BATCH_SIZE = 500
MEMORY_SIZE = 1000000

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    env = Environment()
    window = Window()

    agent = Agent(
        gamma=GAMMA,
        epsilon=EPSILON,
        lr=LEARNING_RATE,
        input_dims=env.state_size,
        action_size=env.action_space,
        mem_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        min_epsilon=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        filename= FILE_NAME
        )

    scores = []
    epsilons = []

    for i in range(NUMBER_OF_GAMES):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            pygame.display.set_caption('Deep Q-Learning Snake')
            running = True
            do = True
            while running and not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                window.draw_screen()
                window.draw_snake(env.snake)
                window.draw_cherry(env.cherry)
                pygame.display.update()
                action = agent.choose_action(observation)
                new_observation, reward, done = env.step(action)
                score += reward
                agent.store_transition(observation, action, reward, new_observation, done)
                observation = new_observation
                agent.learn()

        scores.append(score)
        epsilons.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i, ', score: ', score, ', epsilon: ', agent.epsilon)

    agent.save_model()

    with open('analysis/scores.pkl', 'wb') as f:
        pickle.dump(scores, f)
    with open('analysis/epsilons.pkl', 'wb') as f:
        pickle.dump(epsilons, f)