"""
Replay buffer for Critic Actor algorithm
Author: Dmitriy Movchan
 
"""

import random
from collections import deque

class Buffer(object):

    def __init__(self, buffer_size):
        self.size = buffer_size
        self.length = 0
        self.data = deque()

    def get_batch(self, batch_size):
        return random.sample(self.data, min(self.length, batch_size))

    def add(self, experience):
        self.data.append(experience)
        if self.length < self.size:
            self.length += 1
        else:
            self.data.popleft()

    def clear(self):
        self.data = deque()
        self.length = 0
