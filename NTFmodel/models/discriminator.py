"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
from functools import partial
import torch
import torch.nn as nn
from base.modules import ResBlock, ConvBlock, w_norm_dispatch, activ_dispatch


class ProjectionDiscriminator(nn.Module):
    """ Multi-task discriminator """
    def __init__(self, C, n_fonts, n_chars, w_norm='spectral', activ='none'):
        super().__init__()

        self.activ = activ_dispatch(activ)()
        w_norm = w_norm_dispatch(w_norm)
        self.font_emb = w_norm(nn.Embedding(n_fonts, C))
        self.char_emb = w_norm(nn.Embedding(n_chars, C))

    def forward(self, x, font_indice, char_indice):
        x = self.activ(x)
        font_emb = self.font_emb(font_indice)
        char_emb = self.char_emb(char_indice)
        # print('x')
        # print(x.size())
        # print('font_emb')
        # print(font_emb.size())
        # print('char_emb')
        # print(char_emb.size())
        font_out = torch.einsum('bchw,bc->bhw', x.float(), font_emb.float()).unsqueeze(1)
        char_out = torch.einsum('bchw,bc->bhw', x.float(), char_emb.float()).unsqueeze(1)

        return [font_out, char_out]


class Discriminator(nn.Module):
    """
    spectral norm + ResBlock + Multi-task Discriminator (No patchGAN)
    """
    def __init__(self, n_fonts, n_chars):
        super().__init__()
        ConvBlk = partial(ConvBlock, w_norm="spectral", activ="relu", pad_type="zero")
        ResBlk = partial(ResBlock, w_norm="spectral", activ="relu", pad_type="zero", scale_var=False)

        C = 32
        self.feats = nn.ModuleList([ # input image: 1, 128x128
            ConvBlk(1, C, stride=2, activ='none'),  # 32 64x64 (stirde==2)
            ResBlk(C*1, C*2, downsample=True),    # 64 32x32
            ResBlk(C*2, C*4, downsample=True),    # 128 16x16
            ResBlk(C*4, C*8, downsample=True),    # 256 8x8
            ResBlk(C*8, C*16, downsample=False),  # 512 8x8
            ResBlk(C*16, C*16, downsample=False),  # 512 8x8
        ])

        gap_activ = activ_dispatch("relu")
        self.gap = nn.Sequential(
            gap_activ(),
            nn.AdaptiveAvgPool2d(1)
        )
        # predict
        self.projD = ProjectionDiscriminator(C*16, n_fonts, n_chars, w_norm="spectral")

    def forward(self, x, font_indice, char_indice, out_feats='none'):
        # outfeature: used for feature matching loss; 'None' means no outfeature, while 'all' means out feature
        assert out_feats in {'none', 'all'}
        feats = []
        for layer in self.feats:
            x = layer(x)
            feats.append(x)

        x = self.gap(x)  # final features
        ret = self.projD(x, font_indice, char_indice) # ret = [font_out, char_out]

        if out_feats == 'all':
            ret += feats

        ret = tuple(map(lambda i: i.cuda(), ret))
        return ret
