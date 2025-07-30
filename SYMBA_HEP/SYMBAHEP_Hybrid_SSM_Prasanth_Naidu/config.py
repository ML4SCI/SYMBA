from dataclasses import dataclass, asdict
from typing import Optional,List

@dataclass
class ModelConfig:

    # Project & Run Information
    project_name: str
    run_name: str
    model_name: str

    # Directories
    root_dir: str
    data_dir: str

    # Hardware & Training Setup
    device: str
    epochs: int
    training_batch_size: int
    valid_batch_size: int
    num_workers: int

    # Model Architecture
    embedding_size: int
    nhead: int
    num_encoder_layers: int
    num_decoder_layers: int
    # kan_ff_dims: List[int]
    ff_dims: int
    # is_kan: bool
    # is_pre_norm: bool
    
    # Optimization & Regularization
    warmup_ratio: float
    dropout: float
    weight_decay: float
    optimizer_lr: float
    is_constant_lr: bool

    # Sequence Configuration
    src_max_len: int
    tgt_max_len: int
    is_termwise: bool

    # Training Control
    curr_epoch: int
    train_shuffle: bool
    valid_shuffle: bool
    pin_memory: bool
    world_size: int
    resume_best: bool

    # Optional Parameters
    kan_grid_size:Optional[int] = 8
    dtype: Optional[str] = 'bfloat16'
    run_id: Optional[str] = None
    backend: Optional[str] = 'nccl'
    src_voc_size: Optional[int] = None
    tgt_voc_size: Optional[int] = None
    save_freq: Optional[int] = 3
    save_limit: Optional[int] = 3
    seed: Optional[int] = 42
    update_lr: Optional[float] = None
    end_lr: Optional[float] = 1e-6
    clip_grad_norm: Optional[float] = -1
    save_last: Optional[bool] = True
    log_freq: Optional[int] = 50
    test_freq: Optional[int] = 10
    truncate: Optional[bool] = False
    filter_len: Optional[bool] = False
    debug: Optional[bool] = False
    to_replace: bool = False
    index_pool_size: int = 100

    def to_dict(self):
        """Convert dataclass to dictionary."""
        return asdict(self)


@dataclass
class ModelTestConfig:

    # Model name
    model_name: str

    # Directory where data and model checkpoints will be stored
    root_dir: str

    data_dir: str
    # Device for training (e.g., "cuda" for GPU, "cpu")
    device: str

    # Dimensionality of word embeddings
    embedding_size: int

    # Number of attention heads in the transformer model
    nhead: int

    # Number of encoder layers in the transformer model
    num_encoder_layers : int
    num_decoder_layers: int

    # KAN setting
    # kan_ff_dims: List[int]
    # is_kan:bool
    # is_pre_norm: bool


    
    # FFN dims
    ff_dims: int

    # Dropout rate
    dropout: float

    # Maximum length of source and target sequences
    src_max_len: int
    tgt_max_len: int
    is_termwise: bool
    
    kan_grid_size:Optional[int] = 8

    # Size of vocabulary for source and target sequences
    src_voc_size: Optional[int] = None
    tgt_voc_size: Optional[int] = None

    # Seed for reproducibility
    seed: Optional[int] = 42

    # trucate sequences
    truncate: Optional[bool]= False

    # if debug
    debug: Optional[bool] = False
    
    #to replace index and momentum
    to_replace: bool = False

    #token pool sizes
    index_pool_size : int = 100

    dtype: Optional[str] = 'bfloat16'
    
    def to_dict(self):
        return asdict(self)