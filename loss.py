import torch.nn as nn
from torch.nn.modules import TripletMarginLoss


class IntraInterRanking(nn.Module):
    "Intra-Inter Ranking Loss"

    def __init__(self, alpha=0.2, margin=0.1):
        super(IntraInterRanking, self).__init__()
        self.alpha = alpha
        self.intra = TripletMarginLoss(margin=margin)
        self.inter = TripletMarginLoss(margin=margin)

    def forward(self, a, p, n_intra, n_inter):
        intra_loss = self.intra(a, p, n_intra)
        inter_loss = self.inter(a, p, n_inter)
        loss = intra_loss + self.alpha * inter_loss
        return loss, intra_loss, inter_loss


if __name__ == '__main__':
    import torch, random
    B, D = 3, 100
    criterion = IntraInterRanking()
    a = torch.rand(B, D, requires_grad=True)
    x = torch.rand(B, D, requires_grad=True)
    y = torch.rand(B, D, requires_grad=True)
    z = torch.rand(B, D, requires_grad=True)
    a, b, c = criterion(a, x, y, z)