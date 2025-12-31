from dataclasses import dataclass 

@dataclass 
class Config:
    d_model = 512 
    max_seq_len = 300 
    n_heads = 4 
    dropout_ratio = 0.1 
    bias = False 
    n_encoders = 6 
    n_decoders = 6 
    n_experts = 8 
    k = 3 
    epochs = 10
    path = r"D:\DecoderKAN\QED_data\test-flow.csv" 
    index_token_pool_size = 300 
    special_symbols = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]
    to_replace = True 
    batch_size = 64
    lr = 3e-6