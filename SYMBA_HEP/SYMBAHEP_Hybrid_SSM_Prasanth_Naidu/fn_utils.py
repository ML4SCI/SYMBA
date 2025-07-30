import argparse
import random
from datetime import timedelta
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .config import ModelConfig
from .model.model_factory import construct_model
from .model.helpers.mamba import MixerModel, MambaDecoder

from .constants import BOS_IDX, EOS_IDX, PAD_IDX, SPECIAL_SYMBOLS, UNK_IDX, SEP_IDX, T_IDX
from .tokenizer import Tokenizer, Vocab
from .logger import get_logger

logger = get_logger(__name__) 

def create_tokenizer(df, config):
    """Creates a tokenizer and builds source and target vocabularies.

    Args:
        df (pd.DataFrame): Dataset containing text samples.
        config (object): Configuration object.
        index_pool_size (int): Size of the index pool.
        momentum_pool_size (int): Size of the momentum pool.

    Returns:
        tuple: Tokenizer object, source vocab, target vocab, source index-to-string, target index-to-string.
    """
    tokenizer = Tokenizer(df, config.index_pool_size, SPECIAL_SYMBOLS, UNK_IDX, config.to_replace)
    src_vocab = tokenizer.build_src_vocab()
    tgt_vocab = tokenizer.build_tgt_vocab()
    src_vocab = Vocab(src_vocab, SPECIAL_SYMBOLS, BOS_IDX, PAD_IDX, EOS_IDX, UNK_IDX, SEP_IDX, T_IDX)
    tgt_vocab = Vocab(tgt_vocab, SPECIAL_SYMBOLS, BOS_IDX, PAD_IDX, EOS_IDX, UNK_IDX, SEP_IDX, T_IDX)
    return tokenizer, src_vocab, tgt_vocab


def init_distributed_mode(config):
    """Initializes PyTorch distributed training mode."""
    dist.init_process_group(backend=config.backend, timeout=timedelta(minutes=30))


def create_mask(src: torch.Tensor, tgt: torch.Tensor) -> tuple:
    """Creates source/target masks and padding masks for Transformer.

    Args:
        src (torch.Tensor): Source tensor (S, B).
        tgt (torch.Tensor): Target tensor (T, B).
        device (torch.device): Computation device.

    Returns:
        tuple: (src_padding_mask, tgt_padding_mask)
    """

    src_padding_mask = (src != PAD_IDX)
    tgt_padding_mask = (tgt != PAD_IDX)

    return src_padding_mask, tgt_padding_mask


def generate_unique_random_integers(x, start=0, end=3000):
    """Generates `x` unique integers from the range [start, end].

    Args:
        x (int): Number of integers.
        start (int): Range start.
        end (int): Range end.

    Returns:
        list: List of unique random integers.

    Raises:
        ValueError: If x exceeds the number of unique values in the range.
    """
    if x > (end - start + 1):
        raise ValueError("x cannot be greater than the range of unique values available")
    return random.sample(range(start, end), x)


def decode_sequence(toks: List[int], vocab):
    """Decodes a list of token indices into a string.

    Returns:
        str: Decoded string.
    """
    return ''.join(vocab.decode(toks,include_special_tokens=False))


def collate_fn(batch: list) -> tuple:
    """Collates a batch of (src, tgt) pairs into padded tensors.

    Args:
        batch (list): List of (src, tgt) tuples.

    Returns:
        tuple: Padded src and tgt tensors.
    """
    src_batch = [src for src, _ in batch]
    tgt_batch = [tgt for _, tgt in batch]
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch


def calculate_line_params(point1, point2):
    """Calculates slope and intercept of a line from two points.

    Args:
        point1 (tuple): (x1, y1)
        point2 (tuple): (x2, y2)

    Returns:
        tuple: (slope, intercept)

    Raises:
        ValueError: If x1 == x2 (vertical line).
    """
    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2:
        raise ValueError("The x coordinates must differ to define a valid line.")

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b


def get_model(config):
    """Instantiates and initializes the Transformer model.

    Args:
        config (object): Configuration containing model hyperparameters.

    Returns:
        Model: Initialized Transformer model.
    """
    model = construct_model(config)
    
    # model = Model(
    #     config.num_encoder_layers,
    #     config.num_decoder_layers,
    #     config.embedding_size,
    #     config.nhead,
    #     config.src_voc_size,
    #     config.tgt_voc_size,
    #     config.ff_dims,
    #     config.dropout,
    #     config.is_pre_norm,
    #     config.is_kan,
    #     config.kan_ff_dims,
    #     config.kan_grid_size,
    #     config.device
    # )

    # model.apply(lambda m: init_transformer_weights(m, config.is_mamba))
    
    # logger.info("Weights initialized")
    
    logger.info(str(model)) 

    return model

def parse_ff_dims(ff_dims_str):
    return list(map(int, ff_dims_str.split(',')))

def parse_args():
    """Parses command-line arguments for Transformer training configuration."""

    parser = argparse.ArgumentParser(description="Transformer Training Configuration")

    # Project & model details
    parser.add_argument("--project_name", type=str, required=True, help="Project name")
    parser.add_argument("--run_name", type=str, required=True, help="Run name")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")

    # Directory paths
    parser.add_argument("--root_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")

    # Device & training setup
    parser.add_argument("--device", type=str, default="cuda", help='Device: "cuda" or "cpu"')
    parser.add_argument("--epochs", type=int, required=True, help="Total number of epochs")
    parser.add_argument("--training_batch_size", type=int, required=True, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, required=True, help="Batch size for validation")
    parser.add_argument("--num_workers", type=int, required=True, help="Number of data loader workers")

    # Model architecture
    parser.add_argument("--embedding_size", type=int, required=True, help="Word embedding dimension")
    parser.add_argument("--ff_dims", type=int, required=True, help="Hidden layer dimension")
    parser.add_argument("--nhead", type=int, required=True, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, required=True, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, required=True, help="Number of decoder layers")

    # Optimization settings
    parser.add_argument("--warmup_ratio", type=float, required=True, help="Warmup ratio for learning rate")
    parser.add_argument("--dropout", type=float, required=True, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, required=True, help="Weight decay (AdamW)")
    parser.add_argument("--optimizer_lr", type=float, required=True, help="Optimizer learning rate")
    parser.add_argument("--is_constant_lr", action="store_true", help="Use a constant learning rate")

    # Sequence settings
    parser.add_argument("--src_max_len", type=int, required=True, help="Max source sequence length")
    parser.add_argument("--tgt_max_len", type=int, required=True, help="Max target sequence length")
    parser.add_argument("--is_termwise", action="store_true", help="Termwise dataset")

    # Training state
    parser.add_argument("--curr_epoch", type=int, required=True, help="Current epoch (for resuming)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"],
                        help="Data type used for training: float32, float16, or bfloat16")

    # Data loading
    parser.add_argument("--train_shuffle", action="store_true", help="Shuffle training data")
    parser.add_argument("--valid_shuffle", action="store_true", help="Shuffle validation data")
    parser.add_argument("--pin_memory", action="store_true", help="Enable pinned memory for data loading")

    # Distributed training
    parser.add_argument("--world_size", type=int, default=1, help="Number of processes (distributed training)")
    parser.add_argument("--backend", type=str, default="nccl", help="Distributed training backend")
    parser.add_argument("--resume_best", action="store_true", help="Resume best model")
    parser.add_argument("--run_id", type=str, default=None, help="WandB run ID to resume")

    # Vocabulary settings
    parser.add_argument("--src_voc_size", type=int, default=None, help="Source vocabulary size")
    parser.add_argument("--tgt_voc_size", type=int, default=None, help="Target vocabulary size")

    # Checkpointing
    parser.add_argument("--save_freq", type=int, default=3, help="Checkpoint save frequency (epochs)")
    parser.add_argument("--save_last", action="store_true", help="Save the last model")
    parser.add_argument("--save_limit", type=int, default=5, help="Maximum number of saved checkpoints")

    # Logging & debugging
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--update_lr", type=float, default=None, help="Updated learning rate")
    parser.add_argument("--end_lr", type=float, default=1e-6, help="Final learning rate")
    parser.add_argument("--clip_grad_norm", type=float, default=-1, help="Gradient clipping threshold (-1 to disable)")
    parser.add_argument("--log_freq", type=int, default=50, help="Logging frequency (steps)")
    parser.add_argument("--test_freq", type=int, default=10, help="Testing frequency (steps)")
    parser.add_argument("--truncate", action="store_true", help="Truncate sequences")
    parser.add_argument("--filter_len", action="store_true", help="Filter dataset with sequence length")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Tokenizer settings
    parser.add_argument("--to_replace", action="store_true", help="Replace indices")
    parser.add_argument("--index_pool_size", type=int, default=100, help="Index token pool size")

    args = parser.parse_args()

    return args


def create_config_from_args(args):

    return ModelConfig(
        # Project & model details
        project_name=args.project_name,
        run_name=args.run_name,
        model_name=args.model_name,

        # Directory paths
        root_dir=args.root_dir,
        data_dir=args.data_dir,

        # Device & training setup
        device=args.device,
        epochs=args.epochs,
        training_batch_size=args.training_batch_size,
        valid_batch_size=args.valid_batch_size,
        num_workers=args.num_workers,

        # Model architecture
        embedding_size=args.embedding_size,
        ff_dims=args.ff_dims,  
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,

        # Optimization settings
        warmup_ratio=args.warmup_ratio,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        optimizer_lr=args.optimizer_lr,
        is_constant_lr=args.is_constant_lr,

        # Sequence settings
        src_max_len=args.src_max_len,
        tgt_max_len=args.tgt_max_len,
        is_termwise=args.is_termwise,

        # Training state
        curr_epoch=args.curr_epoch,
        dtype=args.dtype,

        # Data loading
        train_shuffle=args.train_shuffle,
        valid_shuffle=args.valid_shuffle,
        pin_memory=args.pin_memory,

        # Distributed training
        world_size=args.world_size,
        backend=args.backend,
        resume_best=args.resume_best,
        run_id=args.run_id,

        # Vocabulary settings
        src_voc_size=args.src_voc_size,
        tgt_voc_size=args.tgt_voc_size,

        # Checkpointing
        save_freq=args.save_freq,
        save_last=args.save_last,
        save_limit=args.save_limit,

        # Logging & debugging
        seed=args.seed,
        update_lr=args.update_lr,
        end_lr=args.end_lr,
        clip_grad_norm=args.clip_grad_norm,
        log_freq=args.log_freq,
        test_freq=args.test_freq,
        truncate=args.truncate,
        filter_len=args.filter_len,
        debug=args.debug,

        # Tokenizer settings
        to_replace=args.to_replace,
        index_pool_size=args.index_pool_size,
    )
