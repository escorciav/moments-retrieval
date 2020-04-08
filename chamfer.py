import collections
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class ChamferDistance(nn.Module):
    "Chamfer distance"
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, video_feat, lang_feat):
        #pairwise distances matrix, shape = [B,Nv,Nl]
        pairwise_dist = self.batch_pairwise_dist(video_feat,lang_feat)

        # Normalization values
        Nv = pairwise_dist.size()[1]
        Nl = pairwise_dist.size()[2]

        # language_to_clip
        mins, _ = pairwise_dist.min(dim=2)
        language_to_clip_dist = mins.sum(dim=1)/Nv

        # clip_to_language
        mins, _ = pairwise_dist.min(dim=1)
        clip_to_language_dist = mins.sum(dim=1)/Nl

        return language_to_clip_dist + clip_to_language_dist

    @staticmethod
    def batch_pairwise_dist(x, y):
        # implement the formula
        # Dij = ||x-y||**2
        # Expanding it as: Dij = ||x||**2 + ||y||**2 - 2<x,y>
        a = (x*x).sum(dim=2,keepdim=True)
        b = (y*y).sum(dim=2,keepdim=True)
        ab = x @ y.transpose(2,1)  #torch.bmm(x,y.transpose(2,1))
        pairwise_dist = a - 2*ab + b.transpose(2,1)
        return pairwise_dist


class DoubleMaskedChamferDistance(ChamferDistance):
    "Chamfer distance with masking on language and video/obj"
    def __init__(self):
        super(DoubleMaskedChamferDistance, self).__init__()

    def forward(self, video_feat, lang_feat, mask_v, mask_l):
        #pairwise distances matrix, shape = [B,Nv,Nl]
        pairwise_dist = self.batch_pairwise_dist(video_feat, lang_feat)
        masked_minv, masked_minl = self.masked_minimum(pairwise_dist, mask_v, mask_l)

        # Normalization values
        Nv = mask_v.sum(dim=1).clamp(min=1)
        Nl = mask_l.sum(dim=1).clamp(min=1)

        # clip_to_language
        clip_to_language_dist = masked_minv.sum(dim=1)/Nv

        # language_to_clip
        language_to_clip_dist = masked_minl.sum(dim=1)/Nl

        return language_to_clip_dist + clip_to_language_dist

    @staticmethod
    def masked_minimum(p, mv, ml):
        neg = 1 - mv.unsqueeze(-1) * ml.unsqueeze(-2)
        masked = p + neg.type_as(p) * p.max()
        minsv,_ = masked.min(dim=2)
        minsl,_ = masked.min(dim=1)
        return minsv * mv, minsl * ml


if __name__ == '__main__':
    '''TBD Write unit test'''