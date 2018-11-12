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
        intra_loss = F.relu(self.margin + (p - n_intra))
        inter_loss = F.relu(self.margin + (p - n_inter))
        if iw_intra is not None:
            intra_loss = intra_loss * iw_intra
        if iw_inter is not None:
            inter_loss = inter_loss * iw_inter
        loss = (self.w_intra * intra_loss.mean() +
                self.w_inter * inter_loss.mean())
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
    x = torch.rand(B, requires_grad=True)
    y = torch.rand(B, requires_grad=True)
    z = torch.rand(B, requires_grad=True)
    a, b, c = criterion(x, y, z)
