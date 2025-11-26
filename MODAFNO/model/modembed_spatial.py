# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Type, List

import torch
from torch import Tensor, nn


class PositionalEmbedding(nn.Module):
    """
    A module for generating positional embeddings based on timesteps.

    Parameters:
    -----------
    num_channels : int
        Number of channels for the embedding.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

        freqs = torch.pi * torch.arange(
            start=1, end=self.num_channels // 2 + 1, dtype=torch.float32
        )
        self.register_buffer("freqs", freqs)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1).outer(self.freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class OneHotEmbedding(nn.Module):
    """
    A module for generating one-hot embeddings based on timesteps.

    Parameters:
    -----------
    num_channels : int
        Number of channels for the embedding.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        ind = torch.arange(num_channels)
        ind = ind.view(1, len(ind))
        self.register_buffer("indices", ind)

    def forward(self, t: Tensor) -> Tensor:
        ind = t * (self.num_channels - 1)
        return torch.clamp(1 - torch.abs(ind - self.indices), min=0)
    

class StaticConvEmbedding(nn.Module):
    """
    Turn any (B, C, H, W) tensor into a flat vector of length `dim`
    using only conv / pool layers.
    It works as follows:
    1. Apply a sequence of conv2d + batchnorm + ReLU + maxpool2d layers
       for feature extraction and down-sampling. Increases the number of channels to "hidden" values below.
    2. Use adaptive average pooling to enforce a desired reduction of the spatial dimensions (independent of H, W).
    3. Use a 1x1 conv to reduce the number of channels to `dim / (h' * w')`, where (h', w') is the output size after pooling.
    4. Flatten the output to a vector of length `dim`.


    Args
    ----
    in_channels : int          – C   (input channels)
    dim         : int          – desired flattened size
    spatial_out : List[int]   – [h', w'] after pooling.
                                  Default [8, 8]
    hidden      : List[int]   – widths of the “body” conv layers
    """
    def __init__(
        self,
        in_channels: int,
        dim: int,
        spatial_out: List[int] = [8, 8],
        hidden: List[int] = [32, 64, 128],
    ):
        super().__init__()

        body = []
        c_prev = in_channels
        for c_next in hidden:
            body += [
                nn.Conv2d(c_prev, c_next, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=int(c_next/8), num_channels=c_next),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),               # halves H and W each time
            ]
            c_prev = c_next

        # Bring H×W → h'×w' deterministically
        body.append(nn.AdaptiveAvgPool2d(spatial_out))

        # Pick output-channels so that c'·h'·w' = dim
        h_out, w_out = spatial_out
        if dim % (h_out * w_out):
            raise ValueError(
                f"dim={dim} is not divisible by h'*w'={h_out*w_out}. "
                "Choose another spatial_out or another dim."
            )
        c_out = dim // (h_out * w_out)
        body.append(nn.Conv2d(c_prev, c_out, kernel_size=1))  # 1×1 conv

        self.net = nn.Sequential(*body)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)              # (B, c', h', w')
        return x.flatten(1)          # (B, dim)


class ModEmbedNet(nn.Module):
    """
    A network that generates a timestep embedding and processes it with an MLP.

    Parameters:
    -----------
    max_time : float, optional
        Maximum input time. The inputs to `forward` is should be in the range [0, max_time].
    dim : int, optional
        The dimensionality of the time embedding.
    depth : int, optional
        The number of layers in the MLP.
    activation_fn:
        The activation function, default GELU.
    method : str, optional
        The embedding method. Either "sinusoidal" (default) or "onehot".
    """

    def __init__(
        self,
        in_channels: int,
        max_time: float = 1.0,
        dim: int = 64,
        depth: int = 1,
        activation_fn: str = "gelu",
        method: str = "static_conv",
        spatial_out: List[int] = [8, 8],
        hidden: List[int] = [32, 64, 128],
    ):
        super().__init__()
        activation_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
        }
        if activation_fn not in activation_map:
            raise ValueError(f"Unsupported activation function: {activation_fn}")
        activation_fn = activation_map[activation_fn]
        
        self.max_time = max_time
        self.method = method
        if method == "onehot":
            self.onehot_embed = OneHotEmbedding(dim)
        elif method == "sinusoidal":
            self.sinusoid_embed = PositionalEmbedding(dim)
        elif method == "static_conv":
            self.static_conv = StaticConvEmbedding(in_channels, dim, spatial_out, hidden)
        else:
            raise ValueError(f"Embedding '{method}' not supported")

        self.dim = dim
        print('Parameters of modembed:')
        print(f"  max_time: {self.max_time}")
        print(f"  dim: {self.dim}")
        print(f"  depth: {depth}")
        print(f"  activation_fn: {activation_fn.__name__}")
        print(f"  method: {self.method}")
        print(f"  spatial_out: {spatial_out}")
        print(f"  hidden: {hidden}")
        

        blocks = []
        for _ in range(depth):
            blocks.extend([nn.Linear(dim, dim), activation_fn()])
        self.mlp = nn.Sequential(*blocks)

    def forward(self, t: Tensor) -> Tensor:
        t = t / self.max_time
        if self.method == "onehot":
            emb = self.onehot_embed(t)
        elif self.method == "sinusoidal":
            print('entered sinusoidal embedding')
            emb = self.sinusoid_embed(t)
        elif self.method == "static_conv":
            emb = self.static_conv(t)

        return self.mlp(emb)
