import torch
import torch.nn as nn
from functools import partial
import torch.nn as nn
from base.modules import ConvBlock, ResBlock
import base.utils as utils

class StyleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        ConvBlk = partial(ConvBlock, norm="in", activ="relu", pad_type="zero")

        C = 32
        self.layers = nn.Sequential(
            ConvBlk(1, C, 3, 1, 1, norm='none', activ='none'),
            ConvBlk(C*1, C*2, 3, 1, 1, downsample=True),
            ConvBlk(C*2, C*2, 3, 1, 1, downsample=False),
            ConvBlk(C*2, C*4, 3, 1, 1, downsample=True),
            ConvBlk(C*4, C*4, 3, 1, 1, downsample=False)
        )

    def forward(self, x):
        style_feat = self.layers(x)
        return style_feat

class NeRF_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        ConvBlk = partial(ConvBlock, norm="in", activ="relu", pad_type="zero")
        ResBlk = partial(ResBlock, norm="in", activ="relu", pad_type="zero")

        C = 32
        self.layers = nn.ModuleList([
            ConvBlk(C*8 + 3, C*8, 1, 1, 0, norm="none", activ="none"),
            ResBlk(C*8, C*8, 3, 1),
            ResBlk(C*8, C*8, 3, 1),
            ResBlk(C*8, C*8, 3, 1),
            ConvBlk(C*8, C*4, 3, 1, 1, upsample=True),   # 32x32
            ConvBlk(C*4, C*2, 3, 1, 1, upsample=True),   # 64x64
            ConvBlk(C*2, C*1, 3, 1, 1, upsample=True),   # 128x128
        ])

        self.color = nn.Sequential(nn.Conv2d(C, C, kernel_size=3, stride= 1, padding=1, bias=True),
                                    nn.Conv2d(C, 1,kernel_size=3, stride= 1, padding=1, bias=True)) # 32 128x128 -> 1 128x128
        self.sigma = nn.Sequential(nn.Conv2d(C, C, kernel_size=3, stride= 1, padding=1, bias=True),
                                    nn.Conv2d(C, 1,kernel_size=3, stride= 1, padding=1, bias=True)) # 32 128x128 -> 1 128x128

    def forward(self, last):
        for i, layer in enumerate(self.layers):

            if i == 0:
                last = last
            last = layer(last)
        color = self.color(last)
        sigma = self.sigma(last)
        return color, sigma

class StyEmbMf(nn.Module):
    def __init__(self):
        super().__init__()
        self.interval = 15
        self.weight = (1.0/self.interval) * (1.0 + torch.arange(0, self.interval, device="cuda"))
    def forward(self, stylecode):
        stycode_len = stylecode.square().sum(dim=1).sqrt()
        bs, c, h, w = stylecode.shape
        stylecode = stylecode.view(bs, 1, c, h, w).expand(bs, self.interval, c, h, w).contiguous() # bs, self.interval+1, c
        stylecode = stylecode * self.weight.view(1, self.interval, 1,1,1) # bs, self.interval+1, c
        stylecode = stylecode.view(bs*self.interval, c, h, w) # bs, self.interval+1, c -> bs*(self.interval+1), c

        return stylecode, stycode_len

class CharFeat_TransMF(nn.Module):
    def __init__(self):
        super().__init__()
        self.interval = 15

    def forward(self, char_feat):
        bs, c, h, w = char_feat.shape
        char_feat = char_feat.view(bs,1,c,h,w).expand(bs,self.interval,c,h,w).contiguous()
        char_feat = char_feat.view(bs*self.interval,c,h,w)

        return char_feat

class NeRF_Integr(nn.Module):
    def __init__(self):
        super().__init__()
        self.interval = 15
        self.upsample = nn.Upsample(size=[128,128], mode='bilinear')
    def forward(self, color, sigma, stycode_len):

        out = []
        bs, h, w = stycode_len.shape
        bsf, cf, hf, wf = color.shape
        d_interval =  (1.0 / (self.interval)) * stycode_len
        d_interval = d_interval.view(bs,1,h,w)
        d_interval = self.upsample(d_interval)
        color = color.view(bs, self.interval, 1, hf, wf) 
        sigma = sigma.view(bs, self.interval, 1, hf, wf)
        img0 = color[:,0,:,:,:]
        for i in range(self.interval):
            index = self.interval -1 - i
            colori = color[:,index,:,:,:] # bs,1,128,128
            sigmai = sigma[:,index,:,:,:]
            if i == 0:
                Ti = torch.ones_like(colori)
                temp_sigmai = torch.exp(-sigmai*d_interval)
                Cmap = Ti*(1-temp_sigmai)*colori
                out.append(Cmap)
            elif i == 1:
                tmp_T = temp_sigmai
                Ti = Ti*tmp_T
                temp_sigmai = torch.exp(-sigmai*d_interval)
                Cmap = Ti*(1-temp_sigmai)*colori + Cmap
                out.append(Cmap)
            else:
                tmp_T = temp_sigmai
                Ti = Ti*tmp_T
                temp_sigmai = torch.exp(-sigmai*d_interval)
                Cmap = Ti*(1-temp_sigmai)*colori + Cmap
                out.append(Cmap)

        return out, img0

class MFdecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers1 = NeRF_decoder()
        self.style_emb = StyEmbMf()
        self.CharFeat_Trans = CharFeat_TransMF()
        self.NeRF_Integr = NeRF_Integr()

        self.activation1 = nn.Tanh()
        self.activation2 = nn.Sigmoid()

    def forward(self, style_code1, char_code1):
        
        sty_emb1, stycode_len1 = self.style_emb(style_code1)
        char_feat1 = self.CharFeat_Trans(char_code1) # char_feat1 [bs*(self.interval+1),256,16,16]
        
        feat1 = torch.cat([char_feat1, sty_emb1], dim=1) # skip_feat1 [bs*(self.interval+1),512,16,16]
        color1, sigma1 = self.layers1(feat1)
        color1 = self.activation1(color1) 
        sigma1 = self.activation2(sigma1) 

        out, img0 = self.NeRF_Integr(color1, sigma1, stycode_len1)

        out = out[-1]

        return out, img0

class SingleExpert(nn.Module):
    def __init__(self):
        super().__init__()
        ResBlk = partial(ResBlock, norm="in", activ="relu", pad_type="zero", scale_var=False)

        C = 32
        self.layers = nn.ModuleList([
            ResBlk(C*4, C*4, 3, 1),
            ResBlk(C*4, C*4, 3, 1),
            # CBAM(C*4),
            ResBlk(C*4, C*4, 3, 1),
            ResBlk(C*4, C*8, 3, 1, downsample=True), 
            ResBlk(C*8, C*8),
            # CBAM(C*8),
            ResBlk(C*8, C*8)
        ])
        self.skip_idx = 2
        self.feat_shape = {"last": (C*8, 16, 16), "skip": (C*4, 32, 32)}

        self.adapool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            ResBlk(C*8, C*4, 3, 1),
            ResBlk(C*4, C*2, 3, 1),
            ResBlk(C*2, C, 3, 1), 
            ResBlk(C, 3)
        )


    def forward(self, x):
        ret = {}

        for lidx, layer in enumerate(self.layers):
            x = layer(x)
            if lidx == self.skip_idx:
                skip_layer = x
                ret.update({"skip": x})

        char_code = x # 256,16,16
        style_code = self.proj(x) # 3, 16, 16

        return style_code, char_code

    def get_feat_shape(self):
        return self.feat_shape

class Experts_Single(nn.Module):
    def __init__(self, n_experts):
        super(Experts_Single, self).__init__()
        self.n_experts = n_experts
        self.experts1 = SingleExpert()

    def forward(self, x):
        style_code1, char_code1 = self.experts1(x)
        return style_code1, char_code1

    def get_feat_shape(self):
        return self.experts[0].get_feat_shape()

class NTFGenerator(nn.Module):
    def __init__(self, n_experts, n_emb):
        super().__init__()
        self.style_enc = StyleEncoder()

        self.n_experts = n_experts
        self.experts = Experts_Single(self.n_experts)

        self.fact_blocks = {}
        self.recon_blocks = {}

        self.n_in_style = 3
        self.n_in_char = 3

        self.decoder = MFdecoder()

    def encode(self, styimg, charimg):
        feats = self.style_enc(styimg)
        style_code1, _ = self.experts(feats)
        feats = self.style_enc(charimg)
        _, char_code1 = self.experts(feats)

        style_code1 = self.stycode_mean(style_code1,dim=1)

        char_code1 = self.charcode_mean(char_code1,dim=1)

        return style_code1, char_code1

    def stycode_mean(self, stylecode, dim=1):
        n,c,h,w = stylecode.size()
        stylecode = stylecode.view(int(n/self.n_in_style),self.n_in_style,c, h, w)
        mean_stylecode = torch.mean(stylecode,dim=dim)
        mean_stylecode = mean_stylecode.view(int(n/self.n_in_style),c, h, w) # [bs,3]

        return mean_stylecode

    def charcode_mean(self, charcode, dim=1):
        n,c,h,w = charcode.size()
        charcode = charcode.view(int(n/self.n_in_char),self.n_in_char,c, h, w)
        mean_charcode = torch.mean(charcode,dim=dim)
        mean_charcode = mean_charcode.view(int(n/self.n_in_char),c, h, w) # [bs,3]

        return mean_charcode

    def decode(self,  style_code1, char_code1):
        out, img0 = self.decoder(style_code1, char_code1)
        return out, img0

    def infer_styleencode(self, styimg):
        feats = self.style_enc(styimg)
        style_code1, _ = self.experts(feats)
        return style_code1

    def infer_encoder(self, charimg):
        feats = self.style_enc(charimg)
        _, char_code1 = self.experts(feats)
        char_code1 = self.charcode_mean(char_code1,dim=1)

        return char_code1

    def infer(self, style_imgs, char_imgs):
        style_code1, char_code1 = self.encode(style_imgs.flatten(0, 1), char_imgs.flatten(0, 1))
        
        final_img, img0 = self.decode(style_code1, char_code1)

        return final_img, img0

