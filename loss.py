import torch.nn as nn
import torch.nn.functional as F

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
            inter_loss_ = F.relu(self.margin + (p[k] - n_inter[k]))
            if iw_intra is not None:
                intra_loss_ = intra_loss_ * iw_intra
            if iw_inter is not None:
                inter_loss_ = inter_loss_ * iw_inter
            intra_loss += intra_loss_
            inter_loss += inter_loss_
            loss += (self.w_intra * intra_loss_.mean() +
                     self.w_inter * inter_loss_.mean())
        return loss, intra_loss, inter_loss

if __name__ == '__main__':
    import torch
    # simple test IntraInterMarginLoss
    B = 3
    criterion = IntraInterMarginLoss()
    x = torch.rand(B, requires_grad=True)
    y = torch.rand(B, requires_grad=True)
    z = torch.rand(B, requires_grad=True)
    a, b, c = criterion(x, y, z)
