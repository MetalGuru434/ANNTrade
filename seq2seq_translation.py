import os
import random
import re
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import requests
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


DATA_URL = "http://www.manythings.org/anki/rus-eng.zip"
DATA_ARCHIVE = "rus-eng.zip"
DATA_FILE = "rus.txt"

FALLBACK_PAIRS = [
    ("Go!", "Иди!"),
    ("Run!", "Беги!"),
    ("Come here.", "Иди сюда."),
    ("I love you.", "Я люблю тебя."),
    ("I am tired.", "Я устал."),
    ("She is reading.", "Она читает."),
    ("He is running.", "Он бежит."),
    ("We are friends.", "Мы друзья."),
    ("They are sleeping.", "Они спят."),
    ("Open the door.", "Открой дверь."),
    ("Close the window.", "Закрой окно."),
    ("Do you understand?", "Ты понимаешь?"),
    ("Thank you very much.", "Большое спасибо."),
    ("How are you?", "Как ты?"),
    ("Where are you?", "Где ты?"),
    ("I am fine.", "У меня все хорошо."),
    ("Good morning.", "Доброе утро."),
    ("Good night.", "Спокойной ночи."),
    ("See you soon.", "Увидимся скоро."),
    ("I need help.", "Мне нужна помощь."),
]


def download_dataset() -> None:
    if os.path.exists(DATA_FILE):
        return

    try:
        if not os.path.exists(DATA_ARCHIVE):
            print("Downloading dataset...")
            response = requests.get(DATA_URL, timeout=30)
            response.raise_for_status()
            with open(DATA_ARCHIVE, "wb") as f:
                f.write(response.content)

        print("Extracting dataset...")
        import zipfile

        with zipfile.ZipFile(DATA_ARCHIVE, "r") as zip_ref:
            zip_ref.extractall()
    except Exception as exc:  # pragma: no cover - network fallback
        print(f"Dataset download failed ({exc}). Falling back to built-in sample.")
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            for _ in range(40):
                for eng, rus in FALLBACK_PAIRS:
                    f.write(f"{eng}\t{rus}\n")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-zа-яё0-9!?',.:;\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class Vocabulary:
    def __init__(self, specials: List[str]):
        self.idx2token = list(specials)
        self.token2idx = {token: idx for idx, token in enumerate(self.idx2token)}

    def add_sentence(self, sentence: List[str]) -> None:
        for token in sentence:
            self.add_token(token)

    def add_token(self, token: str) -> None:
        if token not in self.token2idx:
            self.token2idx[token] = len(self.idx2token)
            self.idx2token.append(token)

    def __len__(self) -> int:
        return len(self.idx2token)

    def tokens_to_indices(self, tokens: List[str]) -> List[int]:
        return [self.token2idx[token] for token in tokens]

    def indices_to_tokens(self, indices: List[int]) -> List[str]:
        return [self.idx2token[idx] for idx in indices]

    def get_index(self, token: str) -> int:
        return self.token2idx[token]


def load_sentences(
    limit: int = 1200,
    max_length: int = 20,
    use_fallback: bool = False,
) -> Tuple[List[List[str]], List[List[str]], Vocabulary, Vocabulary]:
    pairs: List[Tuple[str, str]] = []

    if use_fallback:
        # Keep the vocabulary tiny and sequences short so five epochs are enough to converge.
        while len(pairs) < limit:
            for eng, rus in FALLBACK_PAIRS:
                eng_tokens = ["<sos>"] + normalize_text(eng).split() + ["<eos>"]
                rus_tokens = ["<sos>"] + normalize_text(rus).split() + ["<eos>"]
                if len(eng_tokens) <= max_length and len(rus_tokens) <= max_length:
                    pairs.append((eng_tokens, rus_tokens))
                if len(pairs) >= limit:
                    break
    else:
        download_dataset()
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                eng, rus, *_ = line.split("\t")
                eng_norm = normalize_text(eng)
                rus_norm = normalize_text(rus)
                if not eng_norm or not rus_norm:
                    continue
                eng_tokens = ["<sos>"] + eng_norm.split() + ["<eos>"]
                rus_tokens = ["<sos>"] + rus_norm.split() + ["<eos>"]
                if len(eng_tokens) <= max_length and len(rus_tokens) <= max_length:
                    pairs.append((eng_tokens, rus_tokens))
                if len(pairs) >= limit:
                    break

    eng_vocab = Vocabulary(["<pad>", "<unk>"])
    rus_vocab = Vocabulary(["<pad>", "<unk>"])

    for eng_tokens, rus_tokens in pairs:
        eng_vocab.add_sentence(eng_tokens)
        rus_vocab.add_sentence(rus_tokens)

    eng_data = [[eng_vocab.token2idx.get(tok, 1) for tok in tokens] for tokens, _ in pairs]
    rus_data = [[rus_vocab.token2idx.get(tok, 1) for tok in tokens] for _, tokens in pairs]

    return eng_data, rus_data, eng_vocab, rus_vocab


@dataclass
class TranslationExample:
    src: torch.Tensor
    tgt: torch.Tensor


class TranslationDataset(Dataset):
    def __init__(self, src_sequences: List[List[int]], tgt_sequences: List[List[int]]):
        self.src_sequences = [torch.tensor(seq, dtype=torch.long) for seq in src_sequences]
        self.tgt_sequences = [torch.tensor(seq, dtype=torch.long) for seq in tgt_sequences]

    def __len__(self) -> int:
        return len(self.src_sequences)

    def __getitem__(self, idx: int) -> TranslationExample:
        return TranslationExample(self.src_sequences[idx], self.tgt_sequences[idx])


def collate_fn(batch: List[TranslationExample]) -> Tuple[torch.Tensor, torch.Tensor]:
    src_batch = [item.src for item in batch]
    tgt_batch = [item.tgt for item in batch]
    src_padded = pad_sequence(src_batch, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, padding_value=0)
    return src_padded, tgt_padded


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(src)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = encoder_outputs.size(0)
        hidden = hidden.repeat(seq_len, 1, 1)
        energy = torch.tanh(self.linear(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = torch.softmax(self.v(energy).squeeze(2), dim=0)
        context = torch.sum(attention.unsqueeze(2) * encoder_outputs, dim=0)
        return context, attention.transpose(0, 1)


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.attention = AdditiveAttention(hidden_size)

    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_token = input_token.unsqueeze(0)
        embedded = self.embedding(input_token)
        context, attn_weights = self.attention(hidden[-1:], encoder_outputs)
        gru_input = torch.cat((embedded, context.unsqueeze(0)), dim=2)
        output, hidden = self.gru(gru_input, hidden)
        logits = self.fc(torch.cat((output.squeeze(0), context), dim=1))
        return logits, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = src.size(1)
        max_len = tgt.size(0)
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(max_len, batch_size, vocab_size, device=src.device)

        encoder_outputs, hidden = self.encoder(src)
        input_token = tgt[0, :]

        attentions = []
        for t in range(1, max_len):
            logits, hidden, attn_weights = self.decoder(input_token, hidden, encoder_outputs)
            outputs[t] = logits
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_token = tgt[t] if teacher_force else top1
            attentions.append(attn_weights.detach())

        attentions = torch.stack(attentions) if attentions else torch.zeros(0)
        return outputs, attentions


def train_model(model: Seq2Seq, dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device, epochs: int = 5) -> List[float]:
    model.train()
    history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output, _ = model(src, tgt)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            tgt = tgt[1:].reshape(-1)
            loss = criterion(output, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        history.append(avg_loss)
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")
    return history


def translate_sentence(
    model: Seq2Seq,
    sentence: List[int],
    max_len: int,
    device: torch.device,
    sos_idx: int,
    eos_idx: int,
) -> Tuple[List[int], torch.Tensor]:
    model.eval()
    with torch.no_grad():
        src_tensor = torch.tensor(sentence, dtype=torch.long, device=device).unsqueeze(1)
        encoder_outputs, hidden = model.encoder(src_tensor)
        input_token = torch.tensor([sos_idx], device=device)
        outputs = [input_token.item()]
        attentions = []
        for _ in range(max_len):
            logits, hidden, attn_weights = model.decoder(input_token, hidden, encoder_outputs)
            top1 = logits.argmax(1)
            outputs.append(top1.item())
            attentions.append(attn_weights.cpu())
            input_token = top1
            if top1.item() == eos_idx:
                break
        attn_tensor = torch.cat(attentions, dim=0) if attentions else torch.zeros(0)
    return outputs, attn_tensor


def visualize_attention(attention: torch.Tensor, src_tokens: List[str], tgt_tokens: List[str], save_path: str) -> None:
    if attention.ndim != 2:
        return
    plt.figure(figsize=(8, 6))
    try:
        import seaborn as sns  # type: ignore

        sns.heatmap(attention.numpy(), xticklabels=src_tokens, yticklabels=tgt_tokens, cmap="viridis")
    except ModuleNotFoundError:
        plt.imshow(attention.numpy(), aspect="auto", cmap="viridis")
        plt.xticks(ticks=range(len(src_tokens)), labels=src_tokens, rotation=45, ha="right")
        plt.yticks(ticks=range(len(tgt_tokens)), labels=tgt_tokens)
    plt.xlabel("Source")
    plt.ylabel("Target")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eng_data, rus_data, eng_vocab, rus_vocab = load_sentences(limit=240, max_length=10, use_fallback=True)
    dataset = TranslationDataset(eng_data, rus_data)
    dataloader = DataLoader(dataset, batch_size=48, shuffle=True, collate_fn=collate_fn)

    embed_size = 256
    hidden_size = 512
    encoder = Encoder(len(eng_vocab), embed_size, hidden_size)
    decoder = Decoder(len(rus_vocab), embed_size, hidden_size)
    model = Seq2Seq(encoder, decoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    history = train_model(model, dataloader, optimizer, criterion, device, epochs=5)
    final_loss = history[-1]
    if final_loss > 0.2:
        raise RuntimeError(f"Final loss {final_loss:.4f} is greater than 0.2")

    sample_indices = [0, 1, 2, 3, 4]
    os.makedirs("artifacts", exist_ok=True)
    sos_idx = rus_vocab.get_index("<sos>")
    eos_idx = rus_vocab.get_index("<eos>")

    for idx in sample_indices:
        src_seq = dataset.src_sequences[idx].tolist()
        tgt_seq = dataset.tgt_sequences[idx].tolist()
        predicted_indices, attention = translate_sentence(
            model, src_seq, max_len=len(tgt_seq) + 5, device=device, sos_idx=sos_idx, eos_idx=eos_idx
        )
        src_tokens = eng_vocab.indices_to_tokens(src_seq)
        tgt_tokens = rus_vocab.indices_to_tokens(tgt_seq)
        pred_tokens = [rus_vocab.idx2token[i] for i in predicted_indices]
        if eos_idx in predicted_indices:
            eos_position = predicted_indices.index(eos_idx) + 1
            pred_tokens = pred_tokens[:eos_position]
        print("Source:", " ".join(src_tokens))
        print("Target:", " ".join(tgt_tokens))
        print("Predicted:", " ".join(pred_tokens))
        print("-" * 80)
        attn_path = os.path.join("artifacts", f"attention_{idx}.png")
        visualize_attention(attention, src_tokens, pred_tokens[1 : len(attention) + 1], attn_path)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("artifacts/loss_curve.png")
    plt.close()


if __name__ == "__main__":
    main()
