import torch
import math
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    """
    Positional encoding module for transformer architectures.

    Args:
        embed_size (int): The embedding size.
        dropout (float): Dropout rate.
        maxlen (int, optional): Maximum sequence length. Defaults to 5000.
    """

    def __init__(self, embed_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, embed_size, 2)
                        * math.log(10000) / embed_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, embed_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    """
    Token embedding module for transformer architectures.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_size (int): The embedding size.
    """

    def __init__(self, vocab_size: int, embed_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.embed_size)

