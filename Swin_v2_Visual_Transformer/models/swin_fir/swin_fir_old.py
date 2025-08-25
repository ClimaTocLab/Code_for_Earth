import sys
sys.path.append('/Users/simon/Documents/Studium/Master/Masterarbeit/code-downscaling/')

from typing import Optional
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F

import numpy as np

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from models.swin_fir.swin_utils import WindowAttention, DropPath, Mlp, SFB
from models.swin_fir.swin_utils import PatchEmbed, PatchUnEmbed, Upsample, UpsampleOneStep
from models.swin_fir.swin_utils import window_partition, window_reverse

from models.lightning_model_template import LightningModelTemplate, LightningModelTemplateSidechannel, cyclicMSELoss, cyclicPSNR, CharbonnierLoss, grad_loss


import lightning as L
from torchmetrics.image import PeakSignalNoiseRatio
from lightning import LightningModule


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer('attn_mask', attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        h, w = x_size
        b, _, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nw*b, window_size*window_size, c
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, '
                f'window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}')


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == 'SFB':
            self.conv = SFB(dim)
        elif resi_connection == 'HSFB':
            self.conv = SFB(dim, 2)
        elif resi_connection == 'identity':
            self.conv = nn.Identity()

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x


#@ARCH_REGISTRY.register()
class SwinFIR(LightningModelTemplate):
    r""" SwinFIR
        A PyTorch impl of : `SwinFIR: Revisiting the SwinIR with Fast Fourier Convolution and
        Improved Training for Image Super-Resolution`, based on Swin Transformer and Fast Fourier Convolution.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
        dataset_metrics: Dataset metrics for normalization. Default: {}
        learning_rate: Learning rate. Default: 1e-4
        channel_names: Channel names for channelwise metrics. Default: []
    """

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 out_chans=20,
                 out_channels=None,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='SFB',
                 dataset_metrics: dict = {}, 
                 learning_rate: float = 1e-4,
                 loss_function: str = 'mse',
                 weight_decay: float = 0.0,
                 lr_scheduling: bool = False,
                 channel_names: list = [],
                 **kwargs):
        
        super(SwinFIR, self).__init__()
        num_in_ch = in_chans
        self.out_chans = out_chans
        self.channel_names = np.array(channel_names)

        norm_layer=nn.LayerNorm
        
        self.seperate_dataset = ('si10' and 'wdir10' not in self.channel_names) and ('u10' and 'v10' in self.channel_names)
        
        if out_channels is None:
            num_out_ch = out_chans
            self.out_channels = list(range(num_out_ch))
        else:
            num_out_ch = len(out_channels)
            self.out_channels = out_channels

        if loss_function == 'mse':
            self.loss_function = F.mse_loss
        elif loss_function == 'cyclic_loss':
            self.loss_function = cyclicMSELoss(np.where(self.channel_names == 'wdir10'), self.out_chans)
            self.mse_function = cyclicMSELoss(np.where(self.channel_names == 'wdir10'), self.out_chans, norm_data=False)
            self.psnr_cyclic = cyclicPSNR(np.where(self.channel_names == 'wdir10'))
        elif loss_function == 'charbonnier':
            self.loss_function = CharbonnierLoss()
        elif loss_function == 'gradient_loss':
            self.loss_function = grad_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        self.weight_decay = weight_decay
        self.lr_scheduling = lr_scheduling

        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.3014, 0.3152, 0.3094)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

        ### Modified Code ###
        self.save_hyperparameters()

        if len(dataset_metrics.keys()) != 0:
            print(f"Dataset metrics provided. Using provided values.")

            self.register_buffer("variable_mean", torch.tensor(dataset_metrics["variable_mean"].reshape(1, -1, 1, 1)).to(torch.float32))
            self.register_buffer("variable_std", torch.tensor(dataset_metrics["variable_std"].reshape(1, -1, 1, 1)).to(torch.float32))
            self.register_buffer("variable_max", torch.tensor(dataset_metrics["variable_max"].reshape(1, -1, 1, 1)).to(torch.float32))
            self.register_buffer("variable_min", torch.tensor(dataset_metrics["variable_min"].reshape(1, -1, 1, 1)).to(torch.float32))
        else:
            print("No dataset metrics provided. Using default values.")
            self.variable_mean = 0
            self.variable_std = 1
            self.variable_max = None
            self.variable_min = None
        
        self.learning_rate = learning_rate

        self.sync_dist = False

        self.valid_psnr = PeakSignalNoiseRatio()
        self.test_psnr = PeakSignalNoiseRatio()
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        input = x
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x
    


class SwinFIR_sidechannel(LightningModelTemplateSidechannel):
    r""" SwinFIR_sidechannel
        A modified PyTorch impl of : `SwinFIR: Revisiting the SwinIR with Fast Fourier Convolution and
        Improved Training for Image Super-Resolution`, based on Swin Transformer and Fast Fourier Convolution.
        In addition to the original implementation, this model also supports side-channel data (e.g. land-sea mask, orography mask).

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        out_channels: List of output channels. Default: None (Same as in_chans)
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
        dataset_metrics_cerra: Dataset metrics output for normalization. Default: {}
        dataset_metrics_era: Dataset metrics input for normalization. Default: {}
        learning_rate: Learning rate. Default: 1e-4
        channel_names: Channel names for channelwise metrics. Default: []
        num_sidechannels: Number of sidechannels. Default: 2
    """

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 out_chans=20,
                 out_channels=None,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='SFB',
                 dataset_metrics_cerra: dict = {}, 
                 dataset_metrics_era: dict = {}, 
                 learning_rate: float = 1e-4,
                 channel_names: list = [],
                 num_sidechannels: int = 2,
                 embed_dim_sidechannels: int = 32,
                 loss_beta: float = 0.5,
                 weights_from_checkpoint_path: Optional[str] = None,
                 loss_function: str = 'mse',
                 lr_scheduling: bool = False,
                 weight_decay: float = 0.0,
                 full_height_cerra: int = 420,
                 full_width_cerra: int = 700,
                 transform: bool = True,
                 standarized: bool = True,
                 metrics_names: list[str] = None,
                 land_mask_bool: bool = False,
                 **kwargs):
        super(SwinFIR_sidechannel, self).__init__()
        num_in_ch = in_chans
        self.out_chans = out_chans
        self.channel_names = np.array(channel_names)
        
        print('Loss function: ', loss_function)

        self.transform = transform 
        self.standarized = standarized

        self.metrics_names = metrics_names
        self.land_mask_bool = land_mask_bool

        norm_layer=nn.LayerNorm

        self.seperate_dataset = ('si10' and 'wdir10' not in self.channel_names) and ('u10' and 'v10' in self.channel_names)

        if out_channels is None:
            num_out_ch = out_chans
            self.out_channels = list(range(num_out_ch))
        else:
            num_out_ch = len(out_channels)
            self.out_channels = out_channels

        self.full_height_cerra = full_height_cerra
        self.full_width_cerra = full_width_cerra
        
        if loss_function == 'mse':
            self.loss_function = F.mse_loss
        elif loss_function == 'mae':
            self.loss_function = F.l1_loss
        elif loss_function == 'Huber':
            self.loss_function = nn.SmoothL1Loss()
        elif loss_function == 'charbonnier':
            self.loss_function = CharbonnierLoss(beta=1.0)
        elif loss_function == 'gradient_loss':
            self.loss_function = grad_loss
        elif loss_function == 'cyclic_loss':
            self.loss_function = cyclicMSELoss(np.where(self.channel_names == 'wdir10'), self.out_chans)
            self.mse_function = cyclicMSELoss(np.where(self.channel_names == 'wdir10'), self.out_chans, norm_data=False)
            self.psnr_cyclic = cyclicPSNR(np.where(self.channel_names == 'wdir10'))
        ###### Include also SSIM, Perceptual loss.... Also a combination of them
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        self.weight_decay = weight_decay
        self.lr_scheduling = lr_scheduling

        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.3014, 0.3152, 0.3094)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.sidechannel_conv_first = nn.Conv2d(num_sidechannels, embed_dim_sidechannels, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.num_sidechannels = num_sidechannels
        self.embed_dim_sidechannels = embed_dim_sidechannels

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
       
        downsample_layer = []
        for _ in range(np.log2(upscale).astype(int)):
            layer = torch.nn.Conv2d(embed_dim_sidechannels, embed_dim_sidechannels, 3, 2, 1)
            downsample_layer.append(layer)

        self.downsample_sidechannel = torch.nn.Sequential(*downsample_layer)

        self.sidechannel_patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=embed_dim_sidechannels,
                embed_dim=embed_dim_sidechannels,
                norm_layer=norm_layer if self.patch_norm else None)    
                
        self.sidechannel_patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim_sidechannels,
            embed_dim=embed_dim_sidechannels,
            norm_layer=norm_layer if self.patch_norm else None)
        

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            self.absolute_pos_embed_sidechannel = nn.Parameter(torch.zeros(1, num_patches, embed_dim_sidechannels))
            trunc_normal_(self.absolute_pos_embed, std=.02)
            trunc_normal_(self.absolute_pos_embed_sidechannel, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim + embed_dim_sidechannels,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.norm_sidechannel = norm_layer(self.embed_dim_sidechannels)

        # build the last conv layer in deep feature extraction
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.conv_after_body_sidechannel = nn.Conv2d(embed_dim_sidechannels, embed_dim_sidechannels, 3, 1, 1)

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

            self.sidechannel_conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim_sidechannels, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.sidechannel_upsample = Upsample(upscale, num_feat)
            self.sidechannel_conv_last = nn.Conv2d(num_feat, num_sidechannels, 3, 1, 1)


        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
            
            self.upsample_sidechannel = UpsampleOneStep(upscale, embed_dim_sidechannels, num_sidechannels,
                                            (patches_resolution[0], patches_resolution[1]))

        elif self.upsampler == 'nearest+conv':
            raise NotImplementedError("Upsampler 'nearest+conv' not implemented for sidechannel yet.")
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            raise NotImplementedError("Upsampler not implemented for sidechannel yet.")
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

        ### Modified Code ###
        self.save_hyperparameters()

        if len(dataset_metrics_cerra.keys()) != 0:
            print(f"Dataset metrics provided. Using provided values.")
            self.register_buffer("variable_mean_cerra", torch.tensor(np.array(dataset_metrics_cerra["variable_mean_cerra"]).reshape(1, -1, 1, 1)).to(torch.float32))
            self.register_buffer("variable_std_cerra", torch.tensor(np.array(dataset_metrics_cerra["variable_std_cerra"]).reshape(1, -1, 1, 1)).to(torch.float32))
            self.register_buffer("variable_max_cerra", torch.tensor(np.array(dataset_metrics_cerra["variable_max_cerra"]).reshape(1, -1, 1, 1)).to(torch.float32))
            self.register_buffer("variable_min_cerra", torch.tensor(np.array(dataset_metrics_cerra["variable_min_cerra"]).reshape(1, -1, 1, 1)).to(torch.float32))
        else:
            print("No dataset metrics provided. Using default values.")
            self.variable_mean_cerra = 0
            self.variable_std_cerra = 1
            self.variable_max_cerra = None
            self.variable_min_cerra = None

        if len(dataset_metrics_era.keys()) != 0:
            print(f"Dataset metrics provided. Using provided values.")

            self.register_buffer("variable_mean_era", torch.tensor(np.array(dataset_metrics_era["variable_mean_era"]).reshape(1, -1, 1, 1)).to(torch.float32))
            self.register_buffer("variable_std_era", torch.tensor(np.array(dataset_metrics_era["variable_std_era"]).reshape(1, -1, 1, 1)).to(torch.float32))
            self.register_buffer("variable_max_era", torch.tensor(np.array(dataset_metrics_era["variable_max_era"]).reshape(1, -1, 1, 1)).to(torch.float32))
            self.register_buffer("variable_min_era", torch.tensor(np.array(dataset_metrics_era["variable_min_era"]).reshape(1, -1, 1, 1)).to(torch.float32))
        else:
            print("No dataset metrics provided. Using default values.")
            self.variable_mean_era = 0
            self.variable_std_era = 1
            self.variable_max_era = None
            self.variable_min_era = None

        if weights_from_checkpoint_path is not None:
            print(f"Using model weights from checkpoint: {weights_from_checkpoint_path}")
            self.load_state_dict(torch.load(weights_from_checkpoint_path, map_location='cpu')['state_dict'])
        
        self.learning_rate = learning_rate
        self.loss_beta = loss_beta

        self.sync_dist = True if torch.cuda.device_count() > 1 else False

        self.valid_psnr = PeakSignalNoiseRatio()
        self.test_psnr = PeakSignalNoiseRatio()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, x_sidechannel):

        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        
        x_sidechannel = self.sidechannel_patch_embed(x_sidechannel)
        
        if self.ape:
            # raise NotImplementedError("Absolute position embedding not implemented for sidechannel yet.")
            x = x + self.absolute_pos_embed
            x_sidechannel = x_sidechannel + self.absolute_pos_embed_sidechannel

        
        x = self.pos_drop(x)
        x_sidechannel = self.pos_drop(x_sidechannel)

        x = torch.cat((x, x_sidechannel), dim=2)

        for layer in self.layers:
            x = layer(x, x_size)

        x = x[:, :, :-self.embed_dim_sidechannels]
        x_sidechannel = x[:, :, -self.embed_dim_sidechannels:]

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        x_sidechannel = self.norm_sidechannel(x_sidechannel)
        x_sidechannel = self.sidechannel_patch_unembed(x_sidechannel, x_size * self.upscale)

        return x, x_sidechannel

    def forward(self, x):
        input = x

        x, x_sidechannel = x      

        if self.upsampler == 'pixelshuffle':
            
            # for classical SR
            x = self.conv_first(x)
            x_sidechannel = self.sidechannel_conv_first(x_sidechannel)

            x_sidechannel = self.downsample_sidechannel(x_sidechannel)

            x_features, x_features_sidechannel = self.forward_features(x, x_sidechannel)

            x = self.conv_after_body(x_features) + x
            x_sidechannel = self.conv_after_body_sidechannel(x_features_sidechannel) + x_sidechannel

            x = self.conv_before_upsample(x)
            x_sidechannel = self.sidechannel_conv_before_upsample(x_sidechannel)

            x = self.conv_last(self.upsample(x))
            x_sidechannel = self.sidechannel_conv_last(self.sidechannel_upsample(x_sidechannel))

        elif self.upsampler == 'pixelshuffledirect':

            # for lightweight SR
            x = self.conv_first(x)
            x_sidechannel = self.sidechannel_conv_first(x_sidechannel)

            x_sidechannel = self.downsample_sidechannel(x_sidechannel)

            x_features, x_features_sidechannel = self.forward_features(x, x_sidechannel)
            
            x = self.conv_after_body(x_features) + x
            x_sidechannel = self.conv_after_body_sidechannel(x_features_sidechannel) + x_sidechannel

            x = self.upsample(x)
            x_sidechannel = self.upsample_sidechannel(x_sidechannel)
        
        elif self.upsampler == 'nearest+conv':
            raise NotImplementedError("Upsampler 'nearest+conv' not implemented for sidechannel yet.")
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            raise NotImplementedError("Upsampler not implemented for sidechannel yet.")
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

       # x = x / self.img_range + self.mean

        return (x, x_sidechannel)
