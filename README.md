# Bi-LSTM + Reinforcement Learning based Hangman solver
<br>
By: Prakhar Mittal
<br>
F2022B3A70426P
<br>
## Objective:
<br>
To train an agent to play Hangman with an accuracy of 50% or more
<br>
## Constraints:
<br>
1. No n-gram approach
2. Apply Formal ML models
3. 6 Attempts maximum
<br>
## Model Architecture:
<br>
Bi-LSTM + Reinforcement Learning Agent(PPO Algorithm), finetuned with endgame heuristic rules
<br>
## Folder structure:
<br>
Files: Contains 4 python scripts:
1. v0_HangmanEnv.py
2. Hangman.py
3. testing.py
4. bilstm.py
<br>
These files contain the classes and functions developed and the training has been done inside
the testing file (main file)
<br>
## Thesis:
<br>
The idea was that a biLSTM model could understand context better than an RL agent, therefore
if we could train it to predict letters given any state of the hidden word then we could use this as
an additional feature for the agent during its training.
<br>
Theoretically, we can mimic the working of the n-gram or frequency based approaches by
generating training examples that contain chunks of letters, allowing the model to recognize
common sequences, WITHOUT implementing n-gram. This idea proved effective, since the
model could:
<br>
predict the next alphabet with a 18% accuracy and<br>
the correct alphabet would be in its top-3 choices 40% of the time.<br>
(for reference a random guess: 3% accuracy on average)<br>
The reinforcement Agent was given a maskable PPO algorithm, trained over 1M timesteps to
play the game. The reward shaping done is based on dynamic entropy calculation, based on the
dictionary provided. Again, mimicking the edge of precomputed frequency models without using
them.
<br>
This model however required rigorous revision in its ability to play endgame states as it would
always lose out on the last alphabet, often the easiest part of the game. A heuristics based rule
for endgame and finetuning gave us:
<br>
A whopping 53% win rate on validation set (in testing dot py)
<br>
Final accuracy in API calls: 13.2%
<br>
## Setback Reasons:
<br>
The model came really close to solving in almost all games played manually. However it could
not win the final alphabet thus marking it a defeat. Even with the fine tuning and entropy based
heuristics, it seems like the words used for testing were too different, rendering the heuristics
often unlucky, and unusable.
<br>
## Suggestions:
<br>
A deeper understanding is required for overcoming the endgame-failure barrier, otherwise the
model seems to do well.
