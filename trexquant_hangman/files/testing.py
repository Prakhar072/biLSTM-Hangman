#!/usr/bin/env python
# coding: utf-8

# # Bi-LSTM + Reinforcement Learning Based Hangman Agent

# ### Dependencies

# In[1]:


import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.envs.registration import registry
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt
import random
import pickle
from stable_baselines3 import A2C
from sb3_contrib import MaskablePPO 
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
import os
import numpy as np
import torch
import tensorflow as tf
import datetime
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import time

from files.bilstm import (
    char_to_idx, idx_to_char, UNKNOWN_IDX, VOCAB_SIZE,
    HangmanStateDataset, HangmanBiLSTM,
    train_bilstm_supervised, evaluate_bilstm,
    load_bilstm_model, prepare_state, predict_next, demo_predictions
)

from files.v0_HangmanEnv import HangmanEnv


# ## Dataset

# In[2]:


def load_word_list(path, max_word_length):
    with open(path, 'r') as f:
        return [line.strip().lower() for line in f if line.strip() and len(line.strip()) <= max_word_length]


def split_word_list_custom(word_list):
    total = len(word_list)
    split1_end = int(0.85 * total)

    split1 = word_list[:split1_end]               # 50%
    split3 = word_list[split1_end:]               # 20%

    return split1, split3

# max_word_length=15
# word_list = load_word_list("dataset/words_250000_train.txt", max_word_length)
# random.shuffle(word_list)
# split1, split3 = split_word_list_custom(word_list)
# print(len(split1), len(split3))  

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ## Bi-LSTM

# In[3]:


# max_word_length = max_word_length
# num_epochs = 20
# batch_size = 64
# lr = 1e-3
# num_examples_per_word = 50
# add_wrong_guesses = True

# pretrained_path = "bilstm_pretrained.pth"

# model = train_bilstm_supervised(
#     split1,
#     save_path=pretrained_path,
#     max_word_length=max_word_length,
#     num_epochs=num_epochs,
#     batch_size=batch_size,
#     lr=lr,
#     num_examples_per_word=num_examples_per_word,
#     add_wrong_guesses=add_wrong_guesses
# )

# model = load_bilstm_model(
#         pretrained_path=pretrained_path,
#         vocab_size=VOCAB_SIZE,
#         embed_dim=32,
#         hidden_size=64,
#         aux_input=True,
#         max_word_length=max_word_length)

# evaluate_bilstm(split3, pretrained_path)

# examples = [
#         ("apple", {"p"}),
#         ("banana", {"a", "n"}),
#         ("grape", {"g", "r"}),
#         ("orange", {"o", "r", "n"}),
#         ("melon", set())]

# demo_predictions(
#         model=model,
#         examples=examples,
#         char_to_idx=char_to_idx,
#         idx_to_char=idx_to_char,
#         max_word_length=max_word_length,
#         topk=5)


# In[32]:


# ======== Begin appended code for endgame fine-tuning & heuristics =========

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
from files.bilstm import (
    load_bilstm_model, HangmanBiLSTM,
    char_to_idx, idx_to_char, UNKNOWN_IDX, VOCAB_SIZE
)

# 1. Dataset for endgame states: mask all but 1-3 letters
class EndgameStateDataset(Dataset):
    """
    Builds samples where each word has only 1–3 letters hidden (endgame states).
    Each sample: (masked_idxs [L], guessed_vec [26], target_idx 0..25)
    """
    def __init__(self, word_list, max_word_length=6, num_examples_per_word=5):
        self.samples = []
        self.max_word_length = max_word_length
        for word in word_list:
            w = word.lower()
            L = len(w)
            if L > max_word_length or L == 0:
                continue
            unique_letters = list(set(w))
            # Only consider words with at least 2 unique letters so there's something to guess
            if len(unique_letters) < 2:
                continue
            for _ in range(num_examples_per_word):
                # Choose how many letters to hide: 1 to min(3, #unique_letters-1)
                num_hide = random.randint(1, min(3, len(unique_letters)-1))
                # Choose which unique letters to hide
                hide_letters = set(random.sample(unique_letters, num_hide))
                # Revealed set = all other letters
                revealed = set(unique_letters) - hide_letters
                # Build one sample per hidden letter (so target is one of hide_letters)
                for target in hide_letters:
                    # Build masked_idxs and guessed_vec
                    idxs = []
                    for i in range(max_word_length):
                        if i < L:
                            c = w[i]
                            if c in revealed:
                                idxs.append(char_to_idx[c])
                            else:
                                idxs.append(UNKNOWN_IDX)
                        else:
                            idxs.append(0)
                    masked_word_idxs = torch.tensor(idxs, dtype=torch.long)
                    guessed_vec = torch.zeros(26, dtype=torch.float32)
                    for c in revealed:
                        if c in char_to_idx:
                            guessed_vec[char_to_idx[c]-1] = 1.0
                    target_idx = char_to_idx[target] - 1
                    self.samples.append((masked_word_idxs, guessed_vec, target_idx))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def fine_tune_endgame(
    pretrained_path,
    word_list,
    max_word_length=6,
    embed_dim=32,
    hidden_size=64,
    lr=1e-4,
    batch_size=64,
    num_epochs=3,
    num_examples_per_word=5,
    save_path=None
):
    """
    Fine-tune the pretrained BiLSTM on endgame states.
    - pretrained_path: path to existing bilstm_pretrained.pth
    - word_list: full vocabulary of words (length ≤ max_word_length)
    - save_path: if provided, path to save the updated state_dict; if None, overwrite pretrained_path.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model
    model = load_bilstm_model(
        pretrained_path=pretrained_path,
        vocab_size=VOCAB_SIZE,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        aux_input=True,
        max_word_length=max_word_length
    ).to(device)
    model.train()
    # Build endgame dataset
    dataset = EndgameStateDataset(word_list, max_word_length, num_examples_per_word)
    if len(dataset) == 0:
        print("[Endgame Fine-tune] No samples generated; check word_list or max_word_length.")
        return
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for masked_idxs, guessed_vec, target_idx in dataloader:
            masked_idxs = masked_idxs.to(device)           # shape [B, L]
            guessed_vec = guessed_vec.to(device)           # [B,26]
            target_idx = target_idx.to(device)             # [B]
            logits = model(masked_idxs, guessed_vec)       # [B,26]
            loss = loss_fn(logits, target_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * masked_idxs.size(0)
        avg = total_loss / len(dataset)
        print(f"[Endgame Fine-tune] Epoch {epoch+1}/{num_epochs}, Loss: {avg:.4f}")
    # Save
    out_path = save_path if save_path is not None else pretrained_path
    torch.save(model.state_dict(), out_path)
    print(f"[Endgame Fine-tune] Saved fine-tuned model to {out_path}")


# In[44]:


# print("Running endgame fine-tuning...")
# fine_tune_endgame(
#     pretrained_path="bilstm_pretrained.pth",
#     word_list=word_list,
#     max_word_length=max_word_length,
#     embed_dim=32,
#     hidden_size=64,
#     lr=1e-4,
#     batch_size=64,
#     num_epochs=3,
#     num_examples_per_word=10,
#     save_path="bilstm_finetuned_endgame.pth"
# )

# model1 = load_bilstm_model(
#         pretrained_path="bilstm_finetuned_endgame.pth",
#         vocab_size=VOCAB_SIZE,
#         embed_dim=32,
#         hidden_size=64,
#         aux_input=True,
#         max_word_length=max_word_length)

# evaluate_bilstm(split3, "bilstm_finetuned_endgame.pth")

# examples = [
#         ("apple", {"a", "p", "e"}),
#         ("banana", {"a", "n"}),
#         ("grape", {"a", "r", "e"}),
#         ("orange", {"o", "r", "n"}),
#         ("melon", set())]

# demo_predictions(
#         model=model1,
#         examples=examples,
#         char_to_idx=char_to_idx,
#         idx_to_char=idx_to_char,
#         max_word_length=max_word_length,
#         topk=5)


# ## RL Setup & Integration

# In[37]:


if "Hangman-v0" not in registry:
    register(
        id='Hangman-v0',
        entry_point=HangmanEnv,
        nondeterministic=True,
    )


# In[38]:


from stable_baselines3.common.callbacks import CheckpointCallback
import os


# In[39]:


def mask_fn(env: gym.Env) -> np.ndarray:
    """Return the action mask by unwrapping wrappers until base env."""
    current_env = env
    if hasattr(current_env, "envs"):
        current_env = current_env.envs[0]
    while not hasattr(current_env, "action_masks") and hasattr(current_env, "env"):
        current_env = current_env.env
    if not hasattr(current_env, "action_masks"):
        raise AttributeError("Base env does not have action_masks() method")
    mask = current_env.action_masks()
    return mask

def train_sb3(word_list,
              pretrained_path,
              max_word_length=6,
              embed_dim=32, hidden_size=64,
              total_timesteps=500_000,
              tensorboard_log="logs/",
              model_save_path="models/ppo_bilstm"):
    """
    Train MaskablePPO with HangmanFeatureExtractor that uses pretrained BiLSTM,
    and env uses entropy-reduction shaping via supervised_path.
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)

    # Create env with entropy shaping: pass supervised_path=pretrained_path
    env = gym.make("Hangman-v0", render_mode=None, word_list=word_list,
                   supervised_path=pretrained_path, max_word_length=max_word_length)
    env = ActionMasker(env, mask_fn)

    # Feature extractor unchanged (uses pretrained BiLSTM for features) :contentReference[oaicite:11]{index=11}
    class HangmanFeatureExtractorFull(BaseFeaturesExtractor):
        def __init__(self, observation_space, pretrained_model_path,
                     vocab_size=VOCAB_SIZE, embed_dim=32,
                     hidden_size=64, max_word_length=6):
            features_dim = hidden_size*2 + 26 + 1
            super(HangmanFeatureExtractorFull, self).__init__(observation_space,
                                                              features_dim=features_dim)
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.bilstm = nn.LSTM(embed_dim, hidden_size,
                                  batch_first=True, bidirectional=True)
            # Load pretrained weights
            tmp = HangmanBiLSTM(vocab_size=vocab_size, embed_dim=embed_dim,
                                hidden_size=hidden_size, aux_input=True,
                                max_word_length=max_word_length)
            tmp.load_state_dict(torch.load(pretrained_model_path, map_location=device))
            # Copy embed and LSTM weights
            self.embed.weight.data.copy_(tmp.embed.weight.data)
            for name, param in tmp.bilstm.named_parameters():
                getattr(self.bilstm, name).data.copy_(param.data)
            # Freeze or unfreeze as desired:
            for param in self.embed.parameters():
                param.requires_grad = True
            for param in self.bilstm.parameters():
                param.requires_grad = True
            self.max_word_length = max_word_length

        def forward(self, observations):
            masked = observations["masked"]
            position_mask = observations["position_mask"]
            guessed = observations["guessed"]
            attempts = observations["attempts_left_norm"]

            # Build indices for LSTM input
            row_sum = masked.sum(dim=-1) > 0
            argmax = masked.argmax(dim=-1)
            letter_idx = torch.where(
                row_sum,
                argmax + 1,
                torch.full_like(argmax, UNKNOWN_IDX)
            )
            masked_word_idxs = letter_idx * position_mask.to(torch.long)

            # Embed + BiLSTM
            x = self.embed(masked_word_idxs)
            lstm_out, _ = self.bilstm(x)
            # Mask padding positions before pooling
            pos_mask = position_mask.unsqueeze(-1)
            masked_out = lstm_out * pos_mask
            sum_out = masked_out.sum(dim=1)
            lengths = position_mask.sum(dim=1, keepdim=True).clamp(min=1).to(torch.float32)
            pooled = sum_out / lengths

            # Concatenate pooled + guessed vector + attempts scalar
            features = torch.cat([pooled, guessed, attempts], dim=1)
            return features

    policy_kwargs = dict(
        features_extractor_class=HangmanFeatureExtractorFull,
        features_extractor_kwargs=dict(
            pretrained_model_path=pretrained_path,
            vocab_size=VOCAB_SIZE,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            max_word_length=max_word_length
        )
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        device=device,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs
    )

    save_dir = os.path.dirname(model_save_path)
    if save_dir == "":
        save_dir = "."
    os.makedirs(save_dir, exist_ok=True)
    name_prefix = os.path.basename(model_save_path)
    # CheckpointCallback saves model every save_freq calls to env.step()
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=save_dir,
        name_prefix=name_prefix
    )

    # Now train with the callback
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback
    )
    # Finally save the last model
    model.save(model_save_path)
    print(f"Saved final RL model to {model_save_path}")

    # model.learn(total_timesteps=total_timesteps)
    # model.save(model_save_path)
    # print(f"Saved RL model to {model_save_path}")


def evaluate_model(model_path, word_list, num_episodes=10):
    print(f"\nEvaluating model: {model_path}")
    # For evaluation, we can omit entropy shaping or include it:
    env = gym.make("Hangman-v0", render_mode=None, word_list=word_list, max_word_length=max_word_length)
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO.load(model_path, env=env, device=device)

    wins = 0
    total_steps = 0

    for i in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        steps = 0
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(
                obs, 
                action_masks=obs["action_mask"],
            )
            obs, reward, terminated, truncated, _ = env.step(action)
            steps += 1
        total_steps += steps
        # Interpret reward==5 as win if using that convention
        final_masked = "".join(env.unwrapped.game.masked)
        if terminated and final_masked == env.unwrapped.game.word:
            wins += 1

    win_rate = wins / num_episodes
    avg_steps = total_steps / num_episodes
    print(f"Win rate: {win_rate*100:.2f}%")
    print(f"Average steps per game: {avg_steps:.2f}")


def test_sb3_once(word_list, model_path='models/ppo_bilstm'):
    env = gym.make("Hangman-v0", render_mode='human', word_list=word_list, max_word_length=max_word_length)
    env = ActionMasker(env, mask_fn)
    model = MaskablePPO.load(model_path, env=env)

    obs, _ = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(observation=obs, action_masks=obs["action_mask"])
        obs, reward, terminated, truncated, _ = env.step(action)
        time.sleep(0.50)
        masked = env.unwrapped.game.masked
        attempts_left = env.unwrapped.game.attempts_left
        print(f"Current masked: {''.join(masked)} | Attempts left: {attempts_left} | Reward: {reward}")
    word = env.unwrapped.game.word
    result = "Win" if reward > 0 else "Loss"
    print(f"Game Over. Word was: {word} → {result}")


# In[42]:


# train_sb3(
#     word_list=split1,
#     pretrained_path="bilstm_finetuned_endgame.pth",
#     max_word_length=max_word_length,
#     embed_dim=32,
#     hidden_size=64,
#     total_timesteps=1000000,
#     tensorboard_log="logs/",
#     model_save_path="models/ppo_bilstm"
# )
# print("training done")
# results = evaluate_model("models/ppo_bilstm", word_list, num_episodes=200)

# # Optionally test one game
# test_sb3_once(split1, model_path='models/ppo_bilstm')


# In[43]:


# test_sb3_once(split1, model_path='models/ppo_bilstm')


# In[52]:


import re
from collections import Counter

def heuristic_next_guess(masked, guessed_set, word_list):
    """
    Suggests a letter using regex and frequency among matching words.
    masked: list of letters and '_' (e.g., ['s','u','_','o','r','g','a','n','i','c','a','l','l','_'])
    guessed_set: set of already guessed letters
    word_list: list of valid words (e.g., training dictionary)

    Returns: letter to guess (str) or None if not enough info
    """
    pattern = ''.join([c if c != '_' else '.' for c in masked])
    regex = re.compile('^' + pattern + '$')
    candidates = [w for w in word_list if len(w) == len(masked) and regex.match(w)]
    if not candidates:
        return None

    freq = Counter()
    for word in candidates:
        for i, c in enumerate(word):
            if masked[i] == '_' and c not in guessed_set:
                freq[c] += 1

    return freq.most_common(1)[0][0] if freq else None

def evaluate_model_with_heuristics(model_path, word_list, num_episodes=10):
    print(f"\nEvaluating model: {model_path}")
    # For evaluation, we can omit entropy shaping or include it:
    env = gym.make("Hangman-v0", render_mode=None, word_list=word_list, max_word_length=max_word_length)
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO.load(model_path, env=env, device=device)

    wins = 0
    total_steps = 0

    for i in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        steps = 0
        truncated = False
        while not (terminated or truncated):
            masked = env.unwrapped.game.masked
            guessed_set = env.unwrapped.game.guessed

            blanks = masked.count('_')
            if blanks <= 2:
                letter = heuristic_next_guess(masked, guessed_set, word_list)
                if letter:
                    action = ord(letter) - ord('a')
                else:
                    action, _ = model.predict(obs, action_masks=obs["action_mask"])
            else:
                action, _ = model.predict(obs, action_masks=obs["action_mask"])

            obs, reward, terminated, truncated, _ = env.step(action)
            steps += 1
        total_steps += steps
        # Interpret reward==5 as win if using that convention
        final_masked = "".join(env.unwrapped.game.masked)
        if terminated and final_masked == env.unwrapped.game.word:
            wins += 1

    win_rate = wins / num_episodes
    avg_steps = total_steps / num_episodes
    print(f"Win rate: {win_rate*100:.2f}%")
    print(f"Average steps per game: {avg_steps:.2f}")


def test_sb3_once_with_heuristics(word_list, model_path='models/ppo_bilstm', max_word_length=6):
    env = gym.make("Hangman-v0", render_mode='human', word_list=word_list, max_word_length=max_word_length)
    env = ActionMasker(env, mask_fn)
    model = MaskablePPO.load(model_path, env=env)

    obs, _ = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        masked = env.unwrapped.game.masked
        guessed_set = set(env.unwrapped.game.guessed)
        blanks = masked.count('_')

        if blanks <= 2:
            letter = heuristic_next_guess(masked, guessed_set, word_list)
            if letter:
                action = ord(letter) - ord('a')
            else:
                action, _ = model.predict(observation=obs, action_masks=obs["action_mask"])
        else:
            action, _ = model.predict(observation=obs, action_masks=obs["action_mask"])

        obs, reward, terminated, truncated, _ = env.step(action)
        time.sleep(0.50)
        masked = env.unwrapped.game.masked
        attempts_left = env.unwrapped.game.attempts_left
        print(f"Current masked: {''.join(masked)} | Attempts left: {attempts_left} | Reward: {reward}")

    final_masked = "".join(env.unwrapped.game.masked)
    word = env.unwrapped.game.word
    result = "Win" if (terminated and final_masked == word) else "Loss"
    print(f"Game Over. Word was: {word} → {result}")




# evaluate_model_with_heuristics("models/ppo_bilstm", word_list, num_episodes=100)
# # Or:
# test_sb3_once_with_heuristics(split3, model_path="models/ppo_bilstm", max_word_length=max_word_length)


# In[53]:


# test_sb3_once_with_heuristics(split3, model_path="models/ppo_bilstm", max_word_length=max_word_length)


# In[ ]:




