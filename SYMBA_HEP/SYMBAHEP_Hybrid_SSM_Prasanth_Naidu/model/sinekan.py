# import math
# from typing import List, Union
# import torch
# import torch.nn as nn

# def forward_step(i_n, grid_size, A, K, C):
#     ratio = A * grid_size**(-K) + C
#     i_n1 = ratio * i_n
#     return i_n1

# class SineKANLayer(torch.nn.Module):
#     def __init__(self, input_dim, output_dim, device='cuda', grid_size=8, is_first=False, add_bias=True, norm_freq=True):
#         super(SineKANLayer,self).__init__()
#         self.grid_size = grid_size
#         self.device = device
#         self.is_first = is_first
#         self.add_bias = add_bias
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.norm_freq = norm_freq
#         self.A, self.K, self.C = 0.9724108095811765, 0.9884401790754128, 0.999449553483052
        
#         self.grid_norm_factor = (torch.arange(grid_size) + 1).reshape(1, 1, grid_size)

#         if is_first:
#             self.amplitudes = torch.nn.Parameter(torch.empty(output_dim, input_dim, 1).normal_(0, .4) / output_dim  / self.grid_norm_factor)
#         else:
#             self.amplitudes = torch.nn.Parameter(torch.empty(output_dim, input_dim, 1).uniform_(-1, 1) / output_dim  / self.grid_norm_factor)

#         grid_phase = torch.arange(1, grid_size + 1).reshape(1, 1, 1, grid_size) / (grid_size + 1)
#         self.input_phase = torch.linspace(0, math.pi, input_dim).reshape(1, 1, input_dim, 1).to(device)
#         phase = grid_phase.to(device) + self.input_phase

#         if norm_freq:
#             self.freq = torch.nn.Parameter(torch.arange(1, grid_size + 1).float().reshape(1, 1, 1, grid_size) / (grid_size + 1)**(1 - is_first))
#         else:
#             self.freq = torch.nn.Parameter(torch.arange(1, grid_size + 1).float().reshape(1, 1, 1, grid_size))

#         for i in range(1, self.grid_size):
#             phase = forward_step(phase, i, self.A, self.K, self.C)

#         self.register_buffer('phase', phase)

#         if self.add_bias:
#             self.bias  = torch.nn.Parameter(torch.ones(1, output_dim) / output_dim)

#     def forward(self, x):
#         x_shape = x.shape
#         output_shape = x_shape[0:-1] + (self.output_dim,)
#         x = torch.reshape(x, (-1, self.input_dim))
#         x_reshaped = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
#         s = torch.sin(x_reshaped * self.freq + self.phase)
#         y = torch.einsum('ijkl,jkl->ij', s, self.amplitudes)
#         if self.add_bias:
#             y += self.bias
#         y = torch.reshape(y, output_shape)
#         return y

#     def __repr__(self):
#         return (f"{self.__class__.__name__}("
#                 f"{self.input_dim} → {self.output_dim}, "
#                 f"grid_size={self.grid_size}, "
#                 f"norm_freq={self.norm_freq}, "
#                 f"add_bias={self.add_bias})")


# class KANFeedForwardBlock(nn.Module):
#     def __init__(self, in_size: int, ff_dims: List[int], grid_size: int = 8, device: Union[str, int] = 'cuda') -> None:
#         super().__init__()
#         self.ffn = nn.ModuleList()
#         for i, d in enumerate(ff_dims):
#             self.ffn.append(SineKANLayer(
#                 input_dim=in_size,
#                 output_dim=d,
#                 grid_size=grid_size,
#                 device=device,
#                 is_first=(i == 0)
#             ))
#             in_size = d

#     def forward(self, x):
#         for f in self.ffn:
#             x = f(x)
#         return x

#     def __repr__(self):
#         rep = f"{self.__class__.__name__}(\n"
#         rep += "  (ffn): ModuleList(\n"
#         for i, layer in enumerate(self.ffn):
#             rep += f"    ({i}): {repr(layer)}\n"
#         rep += "  )\n)"
#         return rep