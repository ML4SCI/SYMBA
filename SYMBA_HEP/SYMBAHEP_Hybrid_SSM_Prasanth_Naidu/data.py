from torch.utils.data import Dataset
import torch

from .constants import BOS_IDX, PAD_IDX, EOS_IDX
from .logger import get_logger

logger = get_logger(__name__)
class Data(Dataset):
    """
    Custom PyTorch dataset for handling data.

    Args:
        df (DataFrame): DataFrame containing data.
    """

    def __init__(self, df, tokenizer, config, src_vocab, tgt_vocab):
        super(Data, self).__init__()
        self.config = config
        self.tgt_tokenize = tokenizer.tgt_tokenize
        self.src_tokenize = tokenizer.src_tokenize
        self.bos_token = torch.tensor([BOS_IDX], dtype=torch.int64)
        self.eos_token = torch.tensor([EOS_IDX], dtype=torch.int64)
        self.pad_token = torch.tensor([PAD_IDX], dtype=torch.int64)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        if self.config.filter_len:
            df = df[
                (df['sqamp'].str.len() <= self.config.tgt_max_len) &
                (df['amp'].str.len() <= self.config.src_max_len)
            ].reset_index(drop=True)
            logger.info(f"Filtered data size is: {len(df)}")
            
        self.tgt_vals = df['sqamp']
        self.src_vals = df['amp']
        
        

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.src_vals)

    def __getitem__(self, idx):
        """
        Get an item from the dataset at the specified index.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Tuple containing source and target tensors.
        """
        src_tokenized = self.src_tokenize(self.src_vals[idx])
        tgt_tokenized = self.tgt_tokenize(self.tgt_vals[idx])
        src_ids = self.src_vocab.encode(src_tokenized)
        tgt_ids = self.tgt_vocab.encode(tgt_tokenized)

        enc_excess_tokens = self.config.src_max_len - len(src_ids) - 3
        dec_excess_tokens = self.config.tgt_max_len - len(tgt_ids) - 3

        if self.config.truncate:
            if enc_excess_tokens < 0:
                src_ids = src_ids[:self.config.src_max_len-3]
            if dec_excess_tokens < 0:
                tgt_ids = tgt_ids[:self.config.tgt_max_len-3]
        else:
            if enc_excess_tokens < 0 or dec_excess_tokens < 0:
                raise ValueError(f"Sentence is too long \n enc_excess_tokens: {enc_excess_tokens}, dec_excess_tokens: {dec_excess_tokens}")
            
        if self.config.is_termwise:
            src_tensor = torch.cat(
                [
                    torch.tensor(src_ids, dtype=torch.int64),
                    self.pad_token,
                ],
                dim=0,
            )
            tgt_tensor = torch.cat(
                [
                    torch.tensor(tgt_ids, dtype=torch.int64),
                    self.pad_token,

                ],
                dim=0,
            )
        else:
            src_tensor = torch.cat(
                [
                    self.bos_token,
                    torch.tensor(src_ids, dtype=torch.int64),
                    self.eos_token,
                    self.pad_token,
                ],
                dim=0,
            )
            tgt_tensor = torch.cat(
                [
                    self.bos_token,
                    torch.tensor(tgt_ids, dtype=torch.int64),
                    self.eos_token,
                    self.pad_token,

                ],
                dim=0,
            )
            
        return src_tensor, tgt_tensor

    @staticmethod
    def get_data(df_train, df_test, df_valid, config, tokenizer, src_vocab,tgt_vocab):
        """
        Create datasets (train, test, and valid)

        Returns:
            dict: Dictionary containing train, test, and valid datasets.
        """
        train = Data(df_train, tokenizer, config,src_vocab,tgt_vocab)
        test = Data(df_test, tokenizer, config,src_vocab,tgt_vocab) if df_test is not None else None
        valid = Data(df_valid, tokenizer, config,src_vocab,tgt_vocab)

        return {'train': train, 'test': test, 'valid': valid}