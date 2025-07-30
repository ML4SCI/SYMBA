import os
import random

import numpy as np
import pandas as pd
import torch



from .fn_utils import (
    create_config_from_args,
    create_tokenizer,
    init_distributed_mode,
    parse_args
)
from .trainer import Trainer
from .logger import get_logger



def main(config, df_train, df_valid, tokenizer, src_vocab, tgt_vocab):
    logger.info(f"Config:\n{config.to_dict()}")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer = Trainer(config, df_train, df_valid, tokenizer, src_vocab, tgt_vocab)
    trainer.fit()

logger = get_logger(__name__) 

if __name__ == '__main__':
    args = parse_args()
    config = create_config_from_args(args)
    config.world_size = int(os.environ['WORLD_SIZE'])
    config.optimizer_lr *= config.world_size

    init_distributed_mode(config)

    df_train = pd.read_csv(config.data_dir + "train.csv")
    df_test = pd.read_csv(config.data_dir + "test.csv")
    df_valid = pd.read_csv(config.data_dir + "valid.csv")

    df = pd.concat([df_train, df_valid, df_test]).reset_index(drop=True)

    tokenizer, src_vocab, tgt_vocab = create_tokenizer(df,config)

    src_vocab 
    config.src_voc_size = len(src_vocab)
    config.tgt_voc_size = len(tgt_vocab)

    # if config.debug:

        # config.epochs = 1
        # df_train = df_train.sample(1000).reset_index(drop=True)

    logger.info(f"TRAIN SAMPLES: {df_train.shape}")
    logger.info("Data loading complete")

    main(config, df_train, df_valid, tokenizer, src_vocab, tgt_vocab)

    torch.distributed.destroy_process_group()
