import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import TripletMarginLoss
import numpy as np
import math

# Constants
N_PAIR = 'n-pair'
ANGULAR = 'angular'
N_PAIR_ANGULAR = 'n-pair-angular'
MAIN_LOSS_CHOICES = (N_PAIR, ANGULAR, N_PAIR_ANGULAR)

CROSS_ENTROPY = 'cross-entropy'

PROPOSAL_FUNCTIONS = ['IntraInterMarginLoss', 
                      'MILNCELossOriginal', 
                      'NPairLoss',
                      'AngularLoss',
                      'NPairAngularLoss'
                    ]

class IntraInterMarginLoss(nn.Module):
    "Intra-Inter Margin Loss (distance based)"

    def __init__(self, w_intra=0.5, w_inter=0.2, margin=0.1):
        super(IntraInterMarginLoss, self).__init__()
        self.margin = margin
        self.w_intra = w_intra
        self.w_inter = w_inter

    def forward(self, scores, iw_intra=None, iw_inter=None):
        p = scores['p'][0]      
        n_intra = scores['n_intra']
        n_inter = scores['n_inter']
        stream_keys = p.keys()
        loss, intra_loss, inter_loss = 0, 0, 0
        for k in stream_keys:
            intra_loss_ = sum([F.relu(self.margin + (p[k] - n[k])) for n in n_intra])
            inter_loss_ = sum([F.relu(self.margin + (p[k] - n[k])) for n in n_inter])
            if iw_intra is not None:
                intra_loss_ = intra_loss_ * iw_intra
            if iw_inter is not None:
                inter_loss_ = inter_loss_ * iw_inter
            intra_loss += intra_loss_
            inter_loss += inter_loss_
            loss += (self.w_intra * intra_loss_.mean() +
                    self.w_inter * inter_loss_.mean())
        return loss


class MILNCELossOriginal(nn.Module):
    """https://arxiv.org/abs/1912.06430"""
    """ Input: scores"""

    def __init__(self, keys):
        super(MILNCELossOriginal, self).__init__()
        self.keys = keys

    def _unpack(self, distances, key):
        return torch.stack(tuple([-1 * d[key] for d in distances]), dim=1)

    def forward(self, scores): # scores = {p:p, n_intra:n_intra, n_inter:n_inter}
        # `positive` is <f(x), g(y)> and `negative` is <f(x'), g(y')>
        # from the paper where <x, y> is the dot product of x and y
        loss = 0
        scores['p'] = [scores['p'][0]]
        for k in self.keys:
            pos_score     = self._unpack(scores['p'], k)
            n_intra_score = self._unpack(scores['n_intra'], k)
            n_inter_score = self._unpack(scores['n_inter'], k)
            # torch.logsumexp(input, dim, keepdim=False, out=None) TRY TO USE THIS
            x = torch.cat((pos_score, n_intra_score, n_inter_score), dim=1)
            x = x.softmax(1)[:, :len(scores['p'])].sum(1).log()
            # loss += 1 * max(0, scores['emb_avg_L2_norm'][k]-10) - x.mean()
            loss -= x.mean()
        return loss


class MILNCELossCrossentropy(nn.Module):
    "Multiple Instance Learn- ing (MIL) and Noise Contrastive Estimation (NCE)"

    def __init__(self, B, keys, device='cpu'):
        super(MILNCELossCrossentropy, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.B = B
        self.keys = keys
        self.device = device
        self.zeros = torch.zeros(B,dtype=torch.long).to(device)

    def _unpack(self, distances, key):
        return [-1 * d[key] for d in distances]

    def _compute_target(self, p):
        batch_size = p[0][self.keys[0]].shape[0]
        if batch_size == self.B:
            return self.zeros
        else:
            return torch.zeros(batch_size,dtype=torch.long).to(self.device)

    def forward(self, scores): # scores = {p:p, n_intra:n_intra, n_inter:n_inter}
        loss = 0
        zeros_batch = self._compute_target(scores['p'])
        for k in self.keys:
            dist_per_key_tuple = (-1*scores['p'][0][k],) + \
                                 tuple(self._unpack(scores['n_intra'], k)) + \
                                 tuple(self._unpack(scores['n_inter'], k))
            distances = torch.stack(dist_per_key_tuple, dim=1)
            loss += self.loss(distances, zeros_batch)
        return loss


class MILNCELossCrossentropyWithLogits(nn.Module):
    "Multiple Instance Learn- ing (MIL) and Noise Contrastive Estimation (NCE)"

    def __init__(self, B, num_pos, num_neg_intra, num_neg_inter, keys, device='cpu'):
        super(MILNCELossCrossentropyWithLogits, self).__init__()

        self.loss = F.binary_cross_entropy_with_logits
        self.B = B
        self.keys = keys
        self.num_samples = num_pos + num_neg_intra + num_neg_inter
        self.device = device
        self.ancillary = torch.zeros(self.num_samples)
        self.ancillary[:num_pos] = 1
        self.target = self.ancillary.expand(B, self.num_samples).to(device)

    def _unpack(self, distances, key):
        return [-1 * d[key] for d in distances]

    def _compute_target(self, p):
        batch_size = p[0][self.keys[0]].shape[0]
        if batch_size == self.B:
            return self.target
        else:
            return self.ancillary.expand(batch_size, self.num_samples).to(self.device)

    def forward(self, p, n_intra, n_inter):
        loss = 0
        target_batch = self._compute_target(p)
        for k in self.keys:
            dist_per_key_tuple = tuple(self._unpack(p, k)) + \
                                 tuple(self._unpack(n_intra, k)) +\
                                 tuple(self._unpack(n_inter, k))
            distances = torch.stack(dist_per_key_tuple, dim=1)
            loss += self.loss(distances, target_batch)
        return loss, None, None


class BlendedLoss(object):
    def __init__(self, main_loss_type, cross_entropy_flag):
        super(BlendedLoss, self).__init__()
        raise NotImplemented
        self.main_loss_type = main_loss_type
        assert main_loss_type in MAIN_LOSS_CHOICES, "invalid main loss: %s" % main_loss_type

        if self.main_loss_type == N_PAIR:
            self.main_loss_fn = NPairLoss()
        elif self.main_loss_type == ANGULAR:
            self.main_loss_fn = AngularLoss()
        elif self.main_loss_type == N_PAIR_ANGULAR:
            self.main_loss_fn = NPairAngularLoss()
        else:
            raise ValueError

        self.cross_entropy_flag = cross_entropy_flag
        self.lambda_blending = 0
        if cross_entropy_flag:
            self.cross_entropy_loss_fn = nn.CrossEntropyLoss()
            self.lambda_blending = 0.3

    def calculate_loss(self, target, output_embedding, output_cross_entropy=None):
        if target is not None:
            target = (target,)

        loss_dict = {}
        blended_loss = 0
        if self.cross_entropy_flag:
            assert output_cross_entropy is not None, "Outputs for cross entropy loss is needed"

            loss_inputs = self._gen_loss_inputs(target, output_cross_entropy)
            cross_entropy_loss = self.cross_entropy_loss_fn(*loss_inputs)
            blended_loss += self.lambda_blending * cross_entropy_loss
            loss_dict[CROSS_ENTROPY + '-loss'] = [cross_entropy_loss.item()]

        loss_inputs = self._gen_loss_inputs(target, output_embedding)
        main_loss_outputs = self.main_loss_fn(*loss_inputs)
        main_loss = main_loss_outputs[0] if type(main_loss_outputs) in (tuple, list) else main_loss_outputs
        blended_loss += (1-self.lambda_blending) * main_loss
        loss_dict[self.main_loss_type+'-loss'] = [main_loss.item()]

        return blended_loss, loss_dict

    @staticmethod
    def _gen_loss_inputs(target, embedding):
        if type(embedding) not in (tuple, list):
            embedding = (embedding,)
        loss_inputs = embedding
        if target is not None:
            if type(target) not in (tuple, list):
                target = (target,)
            loss_inputs += target
        return loss_inputs


class NPairLoss(nn.Module):
    """
    N-Pair loss
    Sohn, Kihyuk. "Improved Deep Metric Learning with Multi-class N-pair Loss Objective," Advances in Neural Information
    Processing Systems. 2016.
    http://papers.nips.cc/paper/6199-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective
    """

    def __init__(self, l2_reg=0.02):
        raise NotImplemented
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, embeddings, target):
        n_pairs, n_negatives = self.get_n_pairs(target)

        if embeddings.is_cuda:
            n_pairs = n_pairs.cuda()
            n_negatives = n_negatives.cuda()

        anchors = embeddings[n_pairs[:, 0]]    # (n, embedding_size)
        positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]    # (n, n-1, embedding_size)

        losses = self.n_pair_loss(anchors, positives, negatives) \
            + self.l2_reg * self.l2_loss(anchors, positives)

        return losses

    @staticmethod
    def get_n_pairs(labels):
        """
        Get index of n-pairs and n-negatives
        :param labels: label vector of mini-batch
        :return: A tuple of n_pairs (n, 2)
                        and n_negatives (n, n-1)
        """
        labels = labels.cpu().data.numpy()
        n_pairs = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            anchor, positive = np.random.choice(label_indices, 2, replace=False)
            n_pairs.append([anchor, positive])

        n_pairs = np.array(n_pairs)

        n_negatives = []
        for i in range(len(n_pairs)):
            negative = np.concatenate([n_pairs[:i, 1], n_pairs[i+1:, 1]])
            n_negatives.append(negative)

        n_negatives = np.array(n_negatives)

        return torch.LongTensor(n_pairs), torch.LongTensor(n_negatives)

    @staticmethod
    def n_pair_loss(anchors, positives, negatives):
        """
        Calculates N-Pair loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))  # (n, 1, n-1)
        x = torch.sum(torch.exp(x), 2)  # (n, 1)
        loss = torch.mean(torch.log(1+x))
        return loss

    @staticmethod
    def l2_loss(anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]


class AngularLoss(NPairLoss):
    """
    Angular loss
    Wang, Jian. "Deep Metric Learning with Angular Loss," ICCV, 2017
    https://arxiv.org/pdf/1708.01682.pdf
    """

    def __init__(self, l2_reg=0.02, angle_bound=1., lambda_ang=2):
        raise NotImplemented
        super(AngularLoss, self).__init__()
        self.l2_reg = l2_reg
        self.angle_bound = angle_bound
        self.lambda_ang = lambda_ang
        self.softplus = nn.Softplus()

    def forward(self, embeddings, target):
        n_pairs, n_negatives = self.get_n_pairs(target)

        if embeddings.is_cuda:
            n_pairs = n_pairs.cuda()
            n_negatives = n_negatives.cuda()

        anchors = embeddings[n_pairs[:, 0]]  # (n, embedding_size)
        positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]  # (n, n-1, embedding_size)

        losses = self.angular_loss(anchors, positives, negatives, self.angle_bound) \
                 + self.l2_reg * self.l2_loss(anchors, positives)

        return losses

    @staticmethod
    def angular_loss(anchors, positives, negatives, angle_bound=1.):
        """
        Calculates angular loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :param angle_bound: tan^2 angle
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = 4. * angle_bound * torch.matmul((anchors + positives), negatives.transpose(1, 2)) \
            - 2. * (1. + angle_bound) * torch.matmul(anchors, positives.transpose(1, 2))  # (n, 1, n-1)

        # Preventing overflow
        with torch.no_grad():
            t = torch.max(x, dim=2)[0]

        x = torch.exp(x - t.unsqueeze(dim=1))
        x = torch.log(torch.exp(-t) + torch.sum(x, 2))
        loss = torch.mean(t + x)

        return loss


class NPairAngularLoss(AngularLoss):
    """
    Angular loss
    Wang, Jian. "Deep Metric Learning with Angular Loss," ICCV, 2017
    https://arxiv.org/pdf/1708.01682.pdf
    """

    def __init__(self, l2_reg=0.02, angle_bound=1., lambda_ang=2):
        raise NotImplemented
        super(NPairAngularLoss, self).__init__()
        self.l2_reg = l2_reg
        self.angle_bound = angle_bound
        self.lambda_ang = lambda_ang

    def forward(self, embeddings, target):
        n_pairs, n_negatives = self.get_n_pairs(target)

        if embeddings.is_cuda:
            n_pairs = n_pairs.cuda()
            n_negatives = n_negatives.cuda()

        anchors = embeddings[n_pairs[:, 0]]    # (n, embedding_size)
        positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]    # (n, n-1, embedding_size)

        losses = self.n_pair_angular_loss(anchors, positives, negatives, self.angle_bound) \
            + self.l2_reg * self.l2_loss(anchors, positives)

        return losses

    def n_pair_angular_loss(self, anchors, positives, negatives, angle_bound=1.):
        """
        Calculates N-Pair angular loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :param angle_bound: tan^2 angle
        :return: A scalar, n-pair_loss + lambda * angular_loss
        """
        n_pair = self.n_pair_loss(anchors, positives, negatives)
        angular = self.angular_loss(anchors, positives, negatives, angle_bound)

        return (n_pair + self.lambda_ang * angular) / (1+self.lambda_ang)


if __name__ == '__main__':
    import torch

    # simple test IntraInterMarginLoss
    B = 3
    criterion = IntraInterMarginLoss()
    x = {'rgb':torch.rand(B, requires_grad=True)}
    y = {'rgb':torch.rand(B, requires_grad=True)}
    z = {'rgb':torch.rand(B, requires_grad=True)}
    a, b, c = criterion(x, y, z)

    # simple test IntraInterMarginLoss
    B = 3
    criterion = MILNCELossOriginal()
    x = [{'rgb':torch.rand(B, requires_grad=True),'obj':torch.rand(B, requires_grad=True)}]
    y = { 'rgb':torch.rand(B, requires_grad=True),'obj':torch.rand(B, requires_grad=True)}
    z = [{'rgb':torch.rand(B, requires_grad=True),'obj':torch.rand(B, requires_grad=True)}]
    a = criterion(x, y, z)
    print(a)
