import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

char_to_idx = {ch: i+1 for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz")}
idx_to_char = {i+1: ch for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz")}
UNKNOWN_IDX = 27
VOCAB_SIZE = 28  

class HangmanStateDataset(Dataset):
    def __init__(self, word_list, max_word_length=6,
                 num_examples_per_word=5, add_wrong_guesses=False, max_wrong=3):
        self.max_word_length = max_word_length
        self.samples = []
        for word in word_list:
            word = word.lower()
            unique_letters = list(set(word))
            for _ in range(num_examples_per_word):
                state_type = random.choices(
                    population=["early", "mid", "late", "chunky"],
                    weights=[0.25, 0.15, 0.25, 0.35], 
                    k=1
                )[0]

                guessed_set = set()

                if state_type == "early":
                    # Reveal 1–2 letters
                    guessed_set = set(random.sample(unique_letters, min(2, len(unique_letters))))

                elif state_type == "mid":
                    # Reveal 40–60% of letters
                    k = max(1, int(0.5 * len(unique_letters)))
                    guessed_set = set(random.sample(unique_letters, min(k, len(unique_letters))))

                elif state_type == "late":
                    # Reveal all but 1 letter
                    k = max(1, len(unique_letters) - 1)
                    guessed_set = set(random.sample(unique_letters, k))

                elif state_type == "chunky":
                    # Force a chunk of the word to be revealed
                    if len(word) >= 3:
                        start = random.randint(0, len(word) - 3)
                        chunk = word[start:start+3]
                        guessed_set = set(chunk)
                        rest = list(set(unique_letters) - guessed_set)
                        guessed_set.update(random.sample(rest, min(2, len(rest))))
                    else:
                        guessed_set = set(random.sample(unique_letters, min(2, len(unique_letters))))

                # Optionally add wrong guesses
                if add_wrong_guesses:
                    wrong_choices = [ch for ch in char_to_idx if ch not in word]
                    num_wrong = random.randint(0, max_wrong)
                    wrong_guessed = random.sample(wrong_choices, min(num_wrong, len(wrong_choices)))
                    guessed_set.update(wrong_guessed)

                # Choose a correct target letter not yet guessed
                remaining = [l for l in unique_letters if l not in guessed_set]
                if not remaining:
                    continue  # all letters revealed
                target = random.choice(remaining)

                self.samples.append((word, guessed_set, target))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        word, revealed, target = self.samples[idx]
        # masked_word_idxs: length max_word_length
        idxs = []
        for i in range(self.max_word_length):
            if i < len(word):
                c = word[i]
                if c in revealed:
                    idxs.append(char_to_idx[c])  # 1–26
                else:
                    idxs.append(UNKNOWN_IDX)     # 27
            else:
                idxs.append(0)  # padding
        masked_word_idxs = torch.tensor(idxs, dtype=torch.long)
        # guessed vector: only revealed here (or include wrong if added)
        guessed_vec = torch.zeros(26, dtype=torch.float32)
        for c in revealed:
            guessed_vec[char_to_idx[c]-1] = 1.0
        target_idx = char_to_idx[target] - 1  # 0–25
        return masked_word_idxs, guessed_vec, target_idx

class HangmanBiLSTM(nn.Module):
    def __init__(self, vocab_size=28, embed_dim=64, hidden_size=128,
                 aux_input=True, max_word_length=12, dropout=0.2):
        super().__init__()
        self.aux_input = aux_input
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        self.fc_input_size = hidden_size * 2
        if aux_input:
            self.fc_input_size += 26

        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, 26)  # output logits for 26 letters

    def forward(self, masked_word_idxs, guessed_letters):
        x = self.embed(masked_word_idxs)  # (batch, L, embed_dim)
        lstm_out, _ = self.bilstm(x)      # (batch, L, hidden*2)
        pooled, _ = torch.max(lstm_out, dim=1)  # (batch, hidden*2)

        if self.aux_input:
            concat = torch.cat([pooled, guessed_letters], dim=1)
        else:
            concat = pooled

        x = self.dropout(concat)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # logits
        return x

def train_bilstm_supervised(word_list, save_path="bilstm_pretrained.pth",
                            max_word_length=12, num_epochs=5, batch_size=64,
                            lr=1e-3, num_examples_per_word=5,
                            add_wrong_guesses=False):
    dataset = HangmanStateDataset(word_list, max_word_length,
                                  num_examples_per_word,
                                  add_wrong_guesses=add_wrong_guesses)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HangmanBiLSTM(vocab_size=VOCAB_SIZE, embed_dim=32,
                          hidden_size=64, aux_input=True,
                          max_word_length=max_word_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for masked_idxs, guessed_vec, target_idx in dataloader:
            masked_idxs = masked_idxs.to(device)
            guessed_vec = guessed_vec.to(device)
            target_idx = target_idx.to(device)
            logits = model(masked_idxs, guessed_vec)  # (batch,26)
            loss = loss_fn(logits, target_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * masked_idxs.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), save_path)
    print("Saved pretrained BiLSTM to", save_path)
    return model

def evaluate_bilstm(val_words,
                      pretrained_path,
                      max_word_length=6,
                      batch_size=64,
                      num_examples_per_word=10,
                      add_wrong_guesses=False,
                      embed_dim=32,
                      hidden_size=64,
                      topk=3):
    """
    Evaluates a pretrained BiLSTM on a validation set.
    Prints top-1 and top-k accuracy.
    """
    val_dataset = HangmanStateDataset(
        val_words,
        max_word_length=max_word_length,
        num_examples_per_word=num_examples_per_word,
        add_wrong_guesses=add_wrong_guesses
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HangmanBiLSTM(
        vocab_size=VOCAB_SIZE,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        aux_input=True,
        max_word_length=max_word_length
    ).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.eval()

    # Top-1 accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for masked_idxs, guessed_vec, target_idx in val_loader:
            masked_idxs = masked_idxs.to(device)
            guessed_vec = guessed_vec.to(device)
            target_idx = target_idx.to(device)
            logits = model(masked_idxs, guessed_vec)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == target_idx).sum().item()
            total += target_idx.size(0)
    accuracy = correct / total if total > 0 else 0.0
    print(f"Validation top-1 accuracy: {accuracy * 100:.2f}%")

    # Top-k accuracy
    correct_topk = 0
    total = 0
    with torch.no_grad():
        for masked_idxs, guessed_vec, target_idx in val_loader:
            masked_idxs = masked_idxs.to(device)
            guessed_vec = guessed_vec.to(device)
            target_idx = target_idx.to(device)
            logits = model(masked_idxs, guessed_vec)
            topk_vals, topk_idxs = torch.topk(logits, k=topk, dim=1)
            target_expand = target_idx.unsqueeze(1).expand(-1, topk)
            matches = (topk_idxs == target_expand).any(dim=1)
            correct_topk += matches.sum().item()
            total += target_idx.size(0)
    accuracy_topk = correct_topk / total if total > 0 else 0.0
    print(f"Validation top-{topk} accuracy: {accuracy_topk * 100:.2f}%")


""" 
other testing functions

"""

def load_bilstm_model(pretrained_path: str,
                      vocab_size: int,
                      embed_dim: int = 32,
                      hidden_size: int = 64,
                      aux_input: bool = True,
                      max_word_length: int = 12) -> HangmanBiLSTM:
    """
    Instantiate HangmanBiLSTM, load state_dict from pretrained_path, set to eval mode.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HangmanBiLSTM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        aux_input=aux_input,
        max_word_length=max_word_length
    ).to(device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def prepare_state(word: str,
                  revealed: set,
                  char_to_idx: dict,
                  max_word_length: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build masked_idxs (shape [1, max_word_length]) and guessed_vec ([1, 26]) for a single word state.
    - word: the target word (string)
    - revealed: set of characters already revealed/guessed correctly
    - char_to_idx: mapping 'a'->1, ..., 'z'->26
    """
    word = word.lower()
    idxs = []
    for i in range(max_word_length):
        if i < len(word):
            c = word[i]
            if c in revealed:
                idxs.append(char_to_idx[c])   # 1..26
            else:
                idxs.append(UNKNOWN_IDX)      # e.g., 27
        else:
            idxs.append(0)  # padding
    masked_idxs = torch.tensor([idxs], dtype=torch.long, device=device)

    guessed_vec = torch.zeros((1, 26), dtype=torch.float32, device=device)
    for c in revealed:
        if c in char_to_idx:
            # char_to_idx[c] is 1..26; we index 0..25
            guessed_vec[0, char_to_idx[c] - 1] = 1.0
    return masked_idxs, guessed_vec


def predict_next(model: HangmanBiLSTM,
                 masked_idxs: torch.Tensor,
                 guessed_vec: torch.Tensor,
                 idx_to_char: dict,
                 topk: int = 5) -> list[tuple[int, str, float]]:
    """
    Given model and a single-state input (masked_idxs [1, L], guessed_vec [1,26]),
    return a list of top-k predictions: (index 0..25, letter, probability).
    """
    with torch.no_grad():
        logits = model(masked_idxs, guessed_vec)  # shape [1,26]
        probs = F.softmax(logits, dim=1)[0]       # shape [26]
        topk_vals, topk_idxs = torch.topk(probs, k=topk)
    results = []
    for prob, idx in zip(topk_vals.tolist(), topk_idxs.tolist()):
        # idx is 0..25; letter index in model is idx+1
        letter_idx = idx + 1
        letter = idx_to_char.get(letter_idx, '?')
        results.append((idx, letter, prob))
    return results


def demo_predictions(model: HangmanBiLSTM,
                     examples: list[tuple[str, set]],
                     char_to_idx: dict,
                     idx_to_char: dict,
                     max_word_length: int = 12,
                     topk: int = 5):
    """
    Run predict_next on a list of (word, revealed_set) examples and print results.
    """
    for word, revealed in examples:
        masked_idxs, guessed_vec = prepare_state(word, revealed, char_to_idx, max_word_length)
        preds = predict_next(model, masked_idxs, guessed_vec, idx_to_char, topk=topk)
        print(f"Word='{word}', Revealed={revealed} -> Top {topk} predictions: {preds}")

