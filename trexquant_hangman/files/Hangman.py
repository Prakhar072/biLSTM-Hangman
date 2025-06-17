#!/usr/bin/env python
# coding: utf-8

# In[13]:


from enum import IntEnum
import random
import numpy as np


# In[14]:


class AgentAction(IntEnum):
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4
    F = 5
    G = 6
    H = 7
    I = 8
    J = 9
    K = 10
    L = 11
    M = 12
    N = 13
    O = 14
    P = 15
    Q = 16
    R = 17
    S = 18
    T = 19
    U = 20
    V = 21
    W = 22
    X = 23
    Y = 24
    Z = 25


# In[16]:


class Hangman:
    def __init__(self, word_list, max_attempts=6):
        self.word_list = word_list
        self.max_attempts = max_attempts
        self.reset()

    def reset(self, seed=None):
        if isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)

        self.word = rng.choice(self.word_list).lower()
        self.masked = ['_'] * len(self.word)
        self.guessed = set()
        self.attempts_left = self.max_attempts
        self.done = False
        self.remaining_actions = set(AgentAction)

    def perform_action(self, action:AgentAction)->int:  # action: letter index (0-25)
        if self.done:
            return 0  # No further action allowed

        if action not in self.remaining_actions:
            return -1

        self.remaining_actions.remove(action)
        letter = chr(65 + action).lower()

        if letter in self.guessed:
            self.attempts_left -= 1
            if self.attempts_left <= 0:
                self.done = True
            return -1

        self.guessed.add(letter)
        if letter in self.word:
            self.masked = [
                letter if self.word[i] == letter else self.masked[i]
                for i in range(len(self.word))
            ]
        else:
            self.attempts_left -= 1

        if self.attempts_left <= 0:
            self.done = True

        if "".join(self.masked) == self.word:
            self.done = True
            return 2  # Correct word guessed

        if letter in self.word:
            return 1
        return -1  # wrong guess or repeat guess

    def render(self):
        print(f"Correct word: {self.word} | Current state: {self.masked} | Attempts left: {self.attempts_left} | Alphabets Guessed: {self.guessed}")
        print()

