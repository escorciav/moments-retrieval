import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import TripletMarginLoss


class IntraInterTripletMarginLoss(nn.Module):
    "Intra-Inter Triplet Margin Loss"

    def __init__(self, w_intra=0.5, w_inter=0.2, margin=0.1):
        super(IntraInterTripletMarginLoss, self).__init__()
        self.intra = TripletMarginLoss(margin=margin)
        self.inter = TripletMarginLoss(margin=margin)
        self.w_intra = w_intra
        self.w_inter = w_inter

    def forward(self, a, p, n_intra, n_inter):
        intra_loss = self.intra(a, p, n_intra)
        inter_loss = self.inter(a, p, n_inter)
        loss = self.w_intra * intra_loss + self.w_inter * inter_loss
        return loss, intra_loss, inter_loss


class IntraInterMarginLoss(nn.Module):
    "Intra-Inter Margin Loss (distance based)"

    def __init__(self, w_intra=0.5, w_inter=0.2, margin=0.1):
        super(IntraInterMarginLoss, self).__init__()
        self.margin = margin
        self.w_intra = w_intra
        self.w_inter = w_inter

    def forward(self, p, n_intra, n_inter, iw_intra=None, iw_inter=None):
        stream_keys = p.keys()
        loss, intra_loss, inter_loss = 0, 0, 0
        for k in stream_keys:
            intra_loss_ = F.relu(self.margin + (p[k] - n_intra[k]))
            # inter_loss_ = F.relu(self.margin + (p[k] - n_inter[k]))
            inter_loss_ = sum([F.relu(self.margin + (p[k] - n[k])) for n in n_inter])
            if iw_intra is not None:
                intra_loss_ = intra_loss_ * iw_intra
            if iw_inter is not None:
                inter_loss_ = inter_loss_ * iw_inter
            intra_loss += intra_loss_
            inter_loss += inter_loss_
            loss += (self.w_intra * intra_loss_.mean() +
                    self.w_inter * inter_loss_.mean())
        return loss, intra_loss, inter_loss


class IntraInterMarginLossRatio(nn.Module):
    "Intra-Inter Margin Loss (distance based)"

    def __init__(self, w_intra=0.5, w_inter=0.2, margin=0.1):
        super(IntraInterMarginLossRatio, self).__init__()
        self.margin = margin
        self.w_intra = w_intra
        self.w_inter = w_inter

    def forward(self, p, n_intra, n_inter, iw_intra=None, iw_inter=None):
        stream_keys = p.keys()
        loss, intra_loss, inter_loss = 0, 0, 0
        for k in stream_keys:
            intra_loss_ = p[k] / n_intra[k].clamp(min=1e-8)
            inter_loss_ = p[k] / n_inter[k].clamp(min=1e-8)
            if iw_intra is not None:
                intra_loss_ = intra_loss_ * iw_intra
            if iw_inter is not None:
                inter_loss_ = inter_loss_ * iw_inter
            intra_loss += intra_loss_
            inter_loss += inter_loss_
            loss += (self.w_intra * intra_loss_.mean() +
                    self.w_inter * inter_loss_.mean())
        return loss, intra_loss, inter_loss


class IntraInterMarginLossPlusClip(IntraInterMarginLoss):
    "Intra-Inter Margin Loss (distance based)"

    def __init__(self, *args, c_intra=0.5, c_inter=0.2, **kwargs):
        super(IntraInterMarginLossPlusClip, self).__init__(*args, **kwargs)
        self.c_intra = c_intra
        self.c_inter = c_inter

    def forward(self, *args):
        p, n_intra, n_inter = args[:3]
        c_p, c_n_intra, c_n_inter = args[3:]
        intra_loss = F.relu(self.margin + (p - n_intra))
        inter_loss = F.relu(self.margin + (p - n_inter))
        c_intra_loss = F.relu(self.margin + (c_p - c_n_intra))
        c_inter_loss = F.relu(self.margin + (c_p - c_n_inter))
        loss = (self.w_intra * intra_loss.mean() +
                self.w_inter * inter_loss.mean() +
                self.c_intra * c_intra_loss.mean() +
                self.c_inter * c_inter_loss.mean())
        return loss, intra_loss, inter_loss


class MILNCELoss(nn.Module):
    "Multiple Instance Learn- ing (MIL) and Noise Contrastive Estimation (NCE)"

    def __init__(self, B, device='cpu'):
        super(MILNCELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.B = B
        self.device = device
        self.zeros = torch.zeros(B,dtype=torch.long).to(device)

    def forward(self, p, n_intra, n_inter):
        keys = list(p.keys())
        batch_size = p[keys[0]].shape[0]
        zeros_batch = self.zeros
        if batch_size != self.B:
            zeros_batch = torch.zeros(batch_size,dtype=torch.long).to(self.device)
        loss = 0
        for k in keys:
            dist_per_key_tuple = (-1*p[k], -1*n_intra[k]) + tuple([-1*ni[k] for ni in n_inter])
            # dist_per_key_tuple = (p[k], n_intra[k], n_inter[k])
            distances = torch.stack(dist_per_key_tuple, dim=1)
            loss += self.loss(distances, zeros_batch)  # None are for compatibility with previous loss output
        return loss, None, None

if __name__ == '__main__':
    import torch

    # simple test IntraInterTripletMarginLoss
    B, D = 3, 100
    criterion = IntraInterTripletMarginLoss()
    a = torch.rand(B, D, requires_grad=True)
    x = torch.rand(B, D, requires_grad=True)
    y = torch.rand(B, D, requires_grad=True)
    z = torch.rand(B, D, requires_grad=True)
    a, b, c = criterion(a, x, y, z)

    # simple test IntraInterMarginLoss
    B = 3
    criterion = IntraInterMarginLoss()
    x = {'rgb':torch.rand(B, requires_grad=True)}
    y = {'rgb':torch.rand(B, requires_grad=True)}
    z = {'rgb':torch.rand(B, requires_grad=True)}
    a, b, c = criterion(x, y, z)

    # simple test IntraInterMarginLoss
    B = 3
    criterion = MILNCELoss()
    x = {'rgb':torch.rand(B, requires_grad=True),'obj':torch.rand(B, requires_grad=True)}
    y = {'rgb':torch.rand(B, requires_grad=True),'obj':torch.rand(B, requires_grad=True)}
    z = {'rgb':torch.rand(B, requires_grad=True),'obj':torch.rand(B, requires_grad=True)}
    a = criterion(x, y, z)
    print(a)
