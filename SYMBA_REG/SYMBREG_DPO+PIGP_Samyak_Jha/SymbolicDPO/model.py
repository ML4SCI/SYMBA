import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
import math

def rightmost_operator_pos(expr_arr):
    binary_operators = [16, 17, 18]
    unary_operators = list(range(19, 47))
    operators = binary_operators + unary_operators

    for i in reversed(range(len(expr_arr))):
        if expr_arr[i] in operators:
            return i

    return -1
def apply_grammar_rules(seq, logits):
    """Applies grammar rules to restrict the next token generation based on the sequence context."""
    if seq.size(1) == 0:
        return logits  # No restrictions for the first token

    last_token = seq[0, -1].item()

    valid_token_mask = torch.zeros_like(logits).fill_(float('-inf'))

    if last_token in range(16, 19):  # Binary operators
        valid_token_mask[:, 3:16] = 0  # Constants
        valid_token_mask[:, 47:58] = 0  # Variables
    elif last_token in range(4, 6):  # s+ or s-
        valid_token_mask[:, 6:16] = 0  # Constants
    else:
        valid_token_mask[:, 3:] = 0  # Default to allowing all except <sos>

    return logits + valid_token_mask

def check_operator_args(seq, operator_pos):
    """Checks if the operator at operator_pos has the required number of arguments."""
    operator_token = seq[0, operator_pos].item()

    if operator_token in range(16, 19):  # Binary operators
        required_args = 2
    elif operator_token in range(19, 47):  # Unary operators
        required_args = 1
    else:
        return False  # Not an operator

    # Count the number of arguments for this operator
    arg_count = 0
    for i in range(operator_pos + 1, seq.size(1)):
        token = seq[0, i].item()
        if token in range(47, 58):  # Variable
            arg_count += 1
        elif token in range(6, 16):  # Constant
            if i > 0 and seq[0, i-1].item() in range(4, 6):  # s+ or s- before constant
                arg_count += 1
        if arg_count == required_args:
            return True  # The operator has enough arguments

    return False  # Not enough arguments

class TokenEmbedding(nn.Module):
    ''' helper Module to convert tensor of input indices into corresponding tensor of token embeddings'''
    
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
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
        self.pos_embedding = torch.zeros((maxlen, emb_size))
        self.pos_embedding[:, 0::2] = torch.sin(pos * den)
        self.pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.pos_embedding = self.pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding_1', self.pos_embedding)

    def forward(self, token_embedding):
        token_embedding = token_embedding.to('cuda')
        self.pos_embedding = self.pos_embedding.to('cuda')
        return self.dropout(token_embedding + self.pos_embedding[:,:token_embedding.size(1), :])

    
class LinearPointEmbedder(nn.Module):
    def __init__(self, vocab_size: int, input_emb_size, emb_size, max_input_points,dropout =0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_emb_size)
        self.emb_size = emb_size
        self.input_size = max_input_points*input_emb_size
        self.fc1 = nn.Linear(self.input_size, emb_size)
        self.fc2 = nn.Linear(emb_size, emb_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        out = self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        bs, n = out.shape[0], out.shape[1]
        out = out.view(bs, n, -1)
        out = self.activation(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    

class Model_seq2seq(nn.Module):
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
                 device : str,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,):
        super(Model_seq2seq, self).__init__()
        self.device = device
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = LinearPointEmbedder(src_vocab_size, input_emb_size, emb_size, max_input_points)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.src_tok_emb(src).to(self.device)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg)).to(self.device)

        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.src_tok_emb(src), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)

    def finish_tokens(self,seq, memory, src_padding_mask, beam_size):
        # Check the rightmost operator and ensure it has enough arguments
        rightmost_op_idx = rightmost_operator_pos(seq[0])
        if rightmost_op_idx is None:
            return seq  # No operator, nothing to finish

        op = seq[0, rightmost_op_idx].item()

        if op in range(16, 19):  # Binary operators
            required_args = 2
        elif op in range(19, 47):  # Unary operators
            required_args = 1
        else :
            return seq
            
            # Not an operator

        current_args = 0
        i = rightmost_op_idx + 1

        while i < seq.size(1) and current_args < required_args:
            token = seq[0, i].item()
            if token in range(4, 6):  # s+ or s- should be followed by a constant (6-15)
                if i + 1 < seq.size(1) and seq[0, i + 1].item() in range(6, 16):
                    current_args += 1
                    i += 2
                    continue
            elif token in range(47, 58) or token == 3:  # Variables
                current_args += 1
            elif token in range(6, 16):  # Constants
                current_args += 1
            i += 1

        while current_args < required_args:
            tgt_mask = (torch.triu(torch.ones((seq.size(1), seq.size(1)), device=self.device))).transpose(0, 1)
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
            tgt_emb = self.positional_encoding(self.tgt_tok_emb(seq))
            out = self.transformer.decoder(tgt_emb, memory, tgt_mask)
            logits = self.generator(out[:, -1, :])

            # Restrict generation to constants and variables (tokens 4-15 and 47-57)
            valid_token_mask = torch.zeros_like(logits).fill_(float('-inf'))
            valid_token_mask[:, 3:16] = 0 
            valid_token_mask[:, 47:58] = 0 
            logits += valid_token_mask

            next_token = torch.argmax(logits, dim=-1)
            if next_token.item() in range(4, 6):
                seq = torch.cat([seq, next_token.unsqueeze(1)], dim=1)
                tgt_mask = (torch.triu(torch.ones((seq.size(1), seq.size(1)), device=self.device))).transpose(0, 1)
                tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
                tgt_emb = self.positional_encoding(self.tgt_tok_emb(seq))
                out = self.transformer.decoder(tgt_emb, memory, tgt_mask)
                logits = self.generator(out[:, -1, :])
                valid_token_mask = torch.zeros_like(logits).fill_(float('-inf'))
                valid_token_mask[:,6:16] = 0
                logits += valid_token_mask
                next_token = torch.argmax(logits, dim=-1)
                seq = torch.cat([seq, next_token.unsqueeze(1)], dim=1)
                current_args += 1
            else:
                seq = torch.cat([seq, next_token.unsqueeze(1)], dim=1)
                current_args += 1
                
        seq = torch.cat([seq, torch.tensor([[58]], device=self.device)], dim=1) 
        return seq
    
    def beam_search(self, src: Tensor, src_mask: Tensor, src_padding_mask: Tensor, beam_size: int = 3, max_len: int = 70, start_state = None):
        # Encode the source sequence
        memory = self.encode(src, src_mask)
        
        # Initialize the decoder input with the <sos> token (assuming 1 is the <sos> token)
        if start_state is None:
            start_symbol = 1  # Modify according to your tokenization scheme
            beam = [(torch.tensor([[start_symbol]], device=self.device), 0)]  # (sequence, score)
        else:
            beam = [(torch.tensor(start_state, device=self.device), 0)]
        completed_sequences = []

        for _ in range(max_len):
            candidates = []
            for seq, score in beam:
                if seq[0, -1].item() == 58:  # Assuming 58 is the <eos> token
                    seq = self.finish_tokens(seq[0, :-1].reshape(1,-1), memory, src_padding_mask, beam_size)
                    completed_sequences.append((seq, score))
                    continue

                # Decode the current sequence
                tgt_mask = (torch.triu(torch.ones((seq.size(1), seq.size(1)), device=self.device))).transpose(0, 1)
                tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
                tgt_emb = self.positional_encoding(self.tgt_tok_emb(seq))
                out = self.transformer.decoder(tgt_emb, memory, tgt_mask)
                logits = self.generator(out[:, -1, :])

                # Apply grammar rules before selecting the next token
                logits = apply_grammar_rules(seq, logits)
                log_probs = torch.log_softmax(logits, dim=-1)

                # Get the top beam_size candidates
                top_log_probs, top_indices = torch.topk(log_probs, beam_size)
                for i in range(beam_size):
                    new_seq = torch.cat([seq, top_indices[:, i].unsqueeze(1)], dim=1)
                    if top_indices[:, i].item() != 58:
                        new_score = score + top_log_probs[:, i].item()

                    # Check if the rightmost operator has the correct number of arguments
                    rightmost_op_pos = rightmost_operator_pos(new_seq[0])
                    if check_operator_args(new_seq, rightmost_op_pos):
                        new_seq = torch.cat([new_seq, torch.tensor([[58]], device=self.device)], dim=1)
                        completed_sequences.append((new_seq, new_score))
                        continue

                    candidates.append((new_seq, new_score))

            # Sort candidates by score and select the top beam_size sequences
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_size]

        # If no sequence ended with <eos>, ensure correctness and return the best candidate
        if not completed_sequences:
            completed_sequences = [(self.finish_tokens(seq, memory, src_padding_mask, beam_size),score) for seq, score in beam]

        # Sort completed sequences by score and return the top ones
        completed_sequences = sorted(completed_sequences, key=lambda x: x[1], reverse=True)

        return completed_sequences[:beam_size]
    
