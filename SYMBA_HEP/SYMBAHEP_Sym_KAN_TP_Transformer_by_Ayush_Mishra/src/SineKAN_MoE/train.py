import os 
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.SineKAN_MoE.tokenizer import SymbolicQEDTokenizer, SymbolicVocab
from src.SineKAN_MoE.config import Config  
from src.SineKAN_MoE.model import Transformer
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import time
import warnings
import torch
from einops import rearrange

class QEDDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        start_time = time.time()
        self.src_vocab = SymbolicVocab(
            tokens=tokenizer.build_src_vocab(),
            special_symbols=tokenizer.special_symbols,
            bos_idx=2,
            pad_idx=0,
            eos_idx=3,
            unk_idx=1,
            sep_idx=4
        )
        self.tgt_vocab = SymbolicVocab(
            tokens=tokenizer.build_tgt_vocab(),
            special_symbols=tokenizer.special_symbols,
            bos_idx=2,
            pad_idx=0,
            eos_idx=3,
            unk_idx=1,
            sep_idx=4
        )
        end_time = time.time()
        print(f"Dataset initialized in {end_time - start_time:.2f} seconds, src_vocab_size: {len(self.src_vocab)}, tgt_vocab_size: {len(self.tgt_vocab)}")
        if len(self.src_vocab) == 5 or len(self.tgt_vocab) == 5:
            warnings.warn("Vocabulary size is minimal (only special tokens). Check dataset or tokenization.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = str(self.data.iloc[idx]["amp"])
        trg = str(self.data.iloc[idx]["sqamp"])
        src_tokens = self.tokenizer.src_tokenize(src)
        trg_tokens = self.tokenizer.tgt_tokenize(trg)
        src_ids = self.src_vocab.encode(src_tokens)
        trg_ids = self.tgt_vocab.encode(trg_tokens)
        src_ids = src_ids[:self.max_length] + [self.src_vocab.pad_idx] * (self.max_length - len(src_ids))
        trg_ids = trg_ids[:self.max_length] + [self.tgt_vocab.pad_idx] * (self.max_length - len(trg_ids))
        return {
            "input_ids": torch.tensor(src_ids, dtype=torch.long),
            "labels": torch.tensor(trg_ids, dtype=torch.long)
        }



def load_data_and_tokenize(path, index_token_pool_size , special_symbols, to_replace, batch_size):
    data_df = pd.read_csv(path)
    tokenizer = SymbolicQEDTokenizer(df=data_df, index_token_pool_size=index_token_pool_size, special_symbols=special_symbols, to_replace=to_replace)
    src_vocab_size = len(tokenizer.build_src_vocab()) + len(special_symbols)
    tgt_vocab_size = len(tokenizer.build_tgt_vocab()) + len(special_symbols)
    dataset = QEDDataset(data_df, tokenizer, index_token_pool_size)
    train_loader = DataLoader(dataset , batch_size = batch_size)
    return train_loader , src_vocab_size , tgt_vocab_size


def token_accuracy(output, target):
    """
    output: (batch, seq_len, vocab_size)
    target: (batch, seq_len)
    """
    preds = torch.argmax(output, dim=-1)       # (batch, seq_len)
    acc = (preds == target).float().mean().item()
    return acc

def sequence_accuracy(output, target):
    """
    output: (batch, seq_len, vocab_size)
    target: (batch, seq_len)
    """
    preds = torch.argmax(output, dim=-1)       # (batch, seq_len)
    correct_sequences = (preds == target).all(dim=1)  # True if all tokens match
    seq_acc = correct_sequences.float().mean().item()
    return seq_acc


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader , src_vocab_size , tgt_vocab_size = load_data_and_tokenize(Config.path , Config.index_token_pool_size , 
                                                                            Config.special_symbols , Config.to_replace , Config.batch_size)
    model = Transformer(src_vocab_size=src_vocab_size , d_model=Config.d_model , tgt_vocab_size=tgt_vocab_size , max_seq_len=Config.max_seq_len , n_heads=Config.n_heads , dropout_ratio=Config.dropout_ratio , bias=Config.bias , 
                        n_decoders=Config.n_decoders , n_encoders=Config.n_encoders , n_experts=Config.n_experts , k=Config.k)
    epochs = Config.epochs
    optimizer = optim.Adam(model.parameters() , lr=Config.lr)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    print("SineKAN MOE Training Started")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_token_acc = 0.0
        total_seq_acc = 0.0

        for batch in train_loader:
            src = batch['input_ids'].to(device)
            target = batch['labels'].to(device)

            optimizer.zero_grad()
            output = model(src, target)  # (batch, seq_len, vocab_size)

            # Flatten for loss
            output_flat = rearrange(output, 'b s c -> (b s) c')
            target_flat = rearrange(target, 'b s -> (b s)')

            # Compute loss
            loss = criterion(output_flat, target_flat)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Compute accuracies
            total_token_acc += token_accuracy(output, target)
            total_seq_acc += sequence_accuracy(output, target)

        avg_loss = total_loss / len(train_loader)
        avg_token_acc = total_token_acc / len(train_loader)
        avg_seq_acc = total_seq_acc / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
            f"Token Acc: {avg_token_acc:.4f} | Seq Acc: {avg_seq_acc:.4f}")


if __name__ == "__main__":
    train()