from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater

class GraphExprDataset(Dataset):
    def __init__(self, items: List[Tuple[Data, torch.Tensor, torch.Tensor]]):
        super().__init__()
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]

def pyg_collate(batch):
    graphs, y_in_list, y_out_list = zip(*batch)
    collater = Collater([], exclude_keys=[])
    data_batch = collater(graphs)
    y_in = torch.nn.utils.rnn.pad_sequence(y_in_list, batch_first=True, padding_value=0)
    y_out = torch.nn.utils.rnn.pad_sequence(y_out_list, batch_first=True, padding_value=0)
    return data_batch, y_in, y_out
