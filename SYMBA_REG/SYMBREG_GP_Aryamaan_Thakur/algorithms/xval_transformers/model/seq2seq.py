import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
import math

# https://github.com/neerajanand321/SYMBA_Pytorch/blob/main/models/seq2seq_transformer.py
class TokenEmbedding(nn.Module):
    ''' helper Module to convert tensor of input indices into corresponding tensor of token embeddings'''
    
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class PositionalEncoding(nn.Module):
    ''' helper Module that adds positional encoding to the token embedding to introduce a notion of word order.'''
    
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])


class xValEmbedder(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
        self.layer_norm = nn.LayerNorm(emb_size)

    def forward(self, tokens, num_array):
        out = self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        out = self.layer_norm(out)
        out *= num_array.unsqueeze(-1)
        return out

class LinearPointEmbedder(nn.Module):
    def __init__(self, vocab_size: int, input_emb_size, emb_size, max_input_points, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_emb_size)
        self.emb_size = emb_size
        self.input_size = max_input_points*input_emb_size
        self.fc1 = nn.Linear(self.input_size, emb_size)
        self.fc2 = nn.Linear(emb_size, emb_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, num_array):
        out = self.embedding(tokens.long()) # * math.sqrt(self.emb_size)
        #dims = torch.tensor(out.size(1)*out.size(2)*out.size(3))
        #mag_norm = 5/torch.sqrt(dims)
        #out += torch.zeros_like(out).uniform_(-mag_norm, mag_norm)
        bs, n = out.shape[0], out.shape[1]
        out *= num_array.unsqueeze(-1)
        out = out.view(bs, n, -1)
        out = self.activation(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class Model(nn.Module):
    '''Seq2Seq Network'''
    
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 input_emb_size: int,
                 max_input_points: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,):
        super(Model, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = LinearPointEmbedder(src_vocab_size, input_emb_size, emb_size, max_input_points, dropout)
        #xValEmbedder(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                num_array: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.src_tok_emb(src, num_array)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))

        
        
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, num_array: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.src_tok_emb(src, num_array), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)
