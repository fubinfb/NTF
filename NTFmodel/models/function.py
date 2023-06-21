import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler



class ResAdaINBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super(ResAdaINBlock, self).__init__()

        self.padding = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, mean_std):
        residual = x
        dim = x.size(1)
        # channel = mean_std.size(1)
        # print('dim')
        # print(channel)

        out = self.padding(x)
        out = self.conv(out)
        out = AdaIN(out, mean_std[:, 0:dim], mean_std[:, dim:dim*2])
        out = self.relu(out)

        out = self.padding(out)
        out = self.conv(out)
        # out = AdaIN(out, mean_std[:, dim*2:dim*3], mean_std[:, dim*3:dim*4])
        out = AdaIN(out, mean_std[:, 0:dim], mean_std[:, dim:dim*2])

        out += residual
        return out

def AdaIN(feature, mean, std, eps=1e-5):
    size = feature.size()
    assert (len(size) == 4)
    N, C = size[:2]
    # size = mean.size()
    # print('feature size')
    # print(size)

    mean = mean.view(N,C,1,1)
    std = std.view(N,C,1,1)

    feature_var = feature.view(N, C, -1).var(dim=2) + eps
    feature_std = feature_var.sqrt().view(N, C, 1, 1)
    feature_mean = feature.view(N, C, -1).mean(dim=2).view(N, C, 1, 1) 

    normalized_feat = (feature - feature_mean.expand(size)) / feature_std.expand(size)
    adain = normalized_feat * std.expand(size) + mean.expand(size)

    return adain