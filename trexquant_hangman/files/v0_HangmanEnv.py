#!/usr/bin/env python
# coding: utf-8

# In[12]:


import gymnasium as gym
from gymnasium import spaces
import files.Hangman as hm
import numpy as np


# In[1]:


#!/usr/bin/env python
# coding: utf-8

import gymnasium as gym
from gymnasium import spaces
import files.Hangman as hm
import numpy as np

import torch
import torch.nn.functional as F
from files.bilstm import load_bilstm_model, prepare_state, UNKNOWN_IDX, char_to_idx
# prepare_state builds (masked_idxs, guessed_vec) for a single state :contentReference[oaicite:1]{index=1}

class HangmanEnv(gym.Env):
    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, word_list, max_attempts=6, max_word_length=6, render_mode=None, supervised_path=None):
        # Keep only words of length ≤ max_word_length
        filtered = [w for w in word_list if len(w) <= max_word_length]
        if len(filtered) < len(word_list):
            print(f"[HangmanEnv] Filtering out {len(word_list) - len(filtered)} words longer than {max_word_length}")
        self.word_list = filtered

        self.max_attempts = max_attempts
        self.render_mode = render_mode
        self.max_word_length = max_word_length
        self.game = hm.Hangman(self.word_list, max_attempts)

        self.action_space = spaces.Discrete(len(hm.AgentAction))
        self.observation_space = spaces.Dict({
            "masked": spaces.Box(low=0, high=1, shape=(self.max_word_length, 26), dtype=np.float32),
            "guessed": spaces.MultiBinary(26),
            "position_mask": spaces.MultiBinary(self.max_word_length),
            "action_mask": spaces.MultiBinary(26),
            "attempts_left_norm": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })

        # Load supervised model if provided
        self.supervised_model = None
        if supervised_path is not None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            try:
                self.supervised_model = load_bilstm_model(
                    pretrained_path=supervised_path,
                    vocab_size=28,
                    embed_dim=32,
                    hidden_size=64,
                    aux_input=True,
                    max_word_length=self.max_word_length
                )
                self.supervised_model.to(self.device).eval()
            except Exception as e:
                print(f"[HangmanEnv] Warning: failed to load supervised model: {e}")
                self.supervised_model = None

    def _get_observation(self):
        masked_matrix = np.zeros((self.max_word_length, 26), dtype=np.float32)
        for i, char in enumerate(self.game.masked):
            if i >= self.max_word_length:
                break
            if char != '_':
                idx = ord(char.lower()) - ord('a')
                masked_matrix[i][idx] = 1.0

        guessed_array = np.zeros(26, dtype=np.float32)
        for letter in self.game.guessed:
            idx = ord(letter) - ord('a')
            guessed_array[idx] = 1.0

        position_mask = np.zeros(self.max_word_length, dtype=np.int8)
        word_len = len(self.game.word)
        for i in range(min(word_len, self.max_word_length)):
            position_mask[i] = 1

        action_mask = 1 - guessed_array
        attempts_left = self.game.attempts_left
        normalized_attempts = attempts_left / self.game.max_attempts

        return {
            "masked": masked_matrix,
            "guessed": guessed_array,
            "position_mask": position_mask,
            "action_mask": action_mask,
            "attempts_left_norm": np.array([normalized_attempts], dtype=np.float32)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        obs = self._get_observation()
        info = {}
        if self.render_mode == 'human':
            self.render()
        return obs, info

    def _compute_entropy(self):
        if self.supervised_model is None:
            return None
        masked_idxs, guessed_vec = prepare_state(
            self.game.word,
            self.game.guessed,
            char_to_idx,
            self.max_word_length
        )
        masked_idxs = masked_idxs.to(self.device)
        guessed_vec = guessed_vec.to(self.device)
        with torch.no_grad():
            logits = self.supervised_model(masked_idxs, guessed_vec)  # shape [1,26]
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy.item()

    def step(self, action):
        if self.game.done:
            obs = self._get_observation()
            word = self.game.word
            masked = "".join(self.game.masked[:self.max_word_length])
            if self.game.attempts_left <= 0 and masked != word:
                return obs, -1.0, True, False, {}
            else:
                return obs, 0.0, True, False, {}

        entropy_before = self._compute_entropy()

        reward = self.game.perform_action(hm.AgentAction(action))
        obs = self._get_observation()
        info = {}

        terminated = ("".join(self.game.masked) == self.game.word)
        truncated = self.game.attempts_left <= 0 and not terminated
        if truncated:
            reward = -1

        entropy_after = self._compute_entropy()
        if (entropy_before is not None) and (entropy_after is not None):
            reduction = entropy_before - entropy_after
            beta = 0.5
            reward = reward + beta * reduction

        if self.render_mode == 'human':
            print(hm.AgentAction(action))
            self.render()

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        obs = self._get_observation()
        return obs["action_mask"].astype(bool)

    def render(self):
        self.game.render()


# In[ ]:




