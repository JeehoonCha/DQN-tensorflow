"""Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import os
import random
import logging
import numpy as np
import cv2

from .utils import save_npy, load_npy

class ReplayMemory:
  def __init__(self, config, model_dir):
    self.model_dir = model_dir

    self.cnn_format = config.cnn_format
    self.memory_size = config.memory_size
    self.actions = np.empty(self.memory_size, dtype=np.uint8)
    self.rewards = np.empty(self.memory_size, dtype=np.integer)
    self.screens = np.empty((self.memory_size, config.screen_height, config.screen_width), dtype=np.float16)
    self.terminals = np.empty(self.memory_size, dtype=np.bool)
    self.history_length = config.history_length
    self.dims = (config.screen_height, config.screen_width)
    self.batch_size = config.batch_size
    self.index = 0
    self.count = 0
    self.current = 0

    # pre-allocate prestates and poststates for minibatch
    self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype=np.float16)
    self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype=np.float16)

  def add(self, screen, reward ,action, terminal):
    screen = np.array(screen, dtype=np.uint8)
    screen = cv2.resize(screen, (84,84), interpolation=cv2.INTER_CUBIC)

    assert screen.shape == self.dims
    # NB! screen is post-state, after action and reward
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.screens[self.current] = screen
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def getState(self, index):
    assert self.count > 0, "replay memory is empty, use at least --random_steps 1"
    if index < 0:
      return self.screens[index + self.count]
    else:
      return self.screens[index % self.count]

  def sample(self):
    # sample random indexes
    indexes = []
    while len(indexes) < self.batch_size:
      index = self.current if self.current != 0 else self.memory_size - 1
      self.prestates[0] = self.getState(index - 1)
      self.poststates[0] = self.getState(index)
      indexes.append(index)

    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]

    if self.cnn_format == 'NHWC':
      return np.transpose(self.prestates, (0, 2, 3, 1)), actions, \
        rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals
    else:
      return self.prestates, actions, rewards, self.poststates, terminals

  def save(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      save_npy(array, os.path.join(self.model_dir, name))

  def load(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      array = load_npy(os.path.join(self.model_dir, name))
