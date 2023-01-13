import torch
from torch import nn
from torch.nn import functional as F

from tqdm import tqdm
import numpy as np
import distributed as dist_fn

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

# applies a sequence of conv3d layers with relu activation for preserving temporal coherence
class Conv3dLatentPostnet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv3d = nn.Sequential(
            self.conv3d_layer(channels=channels),
            self.conv3d_layer(channels=channels),
            self.conv3d_layer(channels=channels, is_final=True)
        )
    
    def conv3d_layer(self, channels=128, kernel_size=3, padding=1, is_final=False):
        if is_final:
            return nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size, padding=padding)
            )
        else:
            return nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size, padding=padding),
                nn.ReLU()
            )
    
    def forward(self, input):
        return self.conv3d(input)

class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
        residual=False,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

        self.residual = residual

        # bottom encoded dimension - batch_size x 128 x 64 x 64
        self.conv3d_encoded_b = Conv3dLatentPostnet(128)
        self.conv3d_encoded_t = Conv3dLatentPostnet(128)

        # self.conv3d_encoded_b = nn.Conv3d(128, 128, 3, padding=1)
        # self.conv3d_encoded_t = nn.Conv3d(128, 128, 3, padding=1)


    def only_encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        return enc_b, enc_t

    def forward(self, input):
        enc_b, enc_t = self.only_encode(input)

        # enc_b dimension -> batch_size x 128 x 64 x 64
        enc_b, enc_t = enc_b.unsqueeze(0).permute(0, 2, 1, 3, 4), enc_t.unsqueeze(0).permute(0, 2, 1, 3, 4)

        # apply the 3d conv on the encoded representations 
        enc_b_conv, enc_t_conv = self.conv3d_encoded_b(enc_b), self.conv3d_encoded_t(enc_t)
        enc_b_conv, enc_t_conv = enc_b_conv.squeeze(0).permute(1, 0, 2, 3), enc_t_conv.squeeze(0).permute(1, 0, 2, 3)

        # generate the quantized representation
        quant_t, quant_b, diff, _, _ = self.encode_quantized(enc_b_conv, enc_t_conv)

        # generate the decoded representation
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode_quantized(self, enc_b, enc_t):
        # enc_b = self.enc_b(input)
        # enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec

# ============================================================================
# ============================= VQVAE BLOB2FULL ==============================
# ============================================================================

class Encode(nn.Module):
    def __init__(
        self,
        in_channel,
        channel,
        n_res_block,
        n_res_channel,
        embed_dim,
        n_embed,
        decay,
        ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)

    def forward(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

class VQVAE_B2F(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        self.encode_face = Encode(in_channel,
            channel,
            n_res_block,
            n_res_channel,
            embed_dim,
            n_embed,
            decay)

        self.encode_rhand = Encode(in_channel,
            channel,
            n_res_block,
            n_res_channel,
            embed_dim,
            n_embed,
            decay)

        self.encode_lhand = Encode(in_channel,
            channel,
            n_res_block,
            n_res_channel,
            embed_dim,
            n_embed,
            decay)

        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, input, save_idx=None, visual_folder=None):
        face, rhand, lhand = input

        quant_t_face, quant_b_face, diff_face, _, _ = self.encode_face(face)
        quant_t_rhand, quant_b_rhand, diff_rhand, _, _ = self.encode_rhand(rhand)
        quant_t_lhand, quant_b_lhand, diff_lhand, _, _ = self.encode_lhand(lhand)

        quant_t = quant_t_face + quant_t_rhand + quant_t_lhand
        quant_b = quant_b_face + quant_b_rhand + quant_b_lhand
        diff = diff_face + diff_rhand + diff_lhand

        dec = self.decode(quant_t, quant_b)

        if save_idx is not None:
            def save_img(img, save_idx, i, dtype):
                img = (img.detach().cpu() + 0.5).numpy()
                img = np.transpose(img, (1,2,0))
                fig = plt.imshow(img, interpolation='nearest')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                plt.savefig('{}/{}_{}_{}.jpg'.\
                    format(visual_folder, save_idx, i, dtype), bbox_inches='tight')

            for i in tqdm(range(min(face.shape[0], 8))):
                save_img(face[i], save_idx, i, 'face')
                save_img(rhand[i], save_idx, i, 'rhand')
                save_img(lhand[i], save_idx, i, 'lhand')
                save_img(x_hat[i], save_idx, i, 'reconstructed')

        return dec, diff

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec