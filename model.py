import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MCN(nn.Module):
    """MCN model
    TODO:
        try max pooling
        (pr): compare runtime with LSTM cell
    """

    def __init__(self, visual_size=4096, lang_size=300, embedding_size=100,
                 dropout=0.3, rec_layers=3, max_length=None):
        super(MCN, self).__init__()
        self.embedding_size = embedding_size
        self.max_length = max_length

        self.img_encoder = nn.Sequential(
          nn.Linear(visual_size, 500),
          nn.ReLU(inplace=True),
          nn.Linear(500, embedding_size),
          nn.Dropout(dropout)
        )

        self.sentence_encoder = nn.LSTM(
            lang_size, 1000, rec_layers, batch_first=True)
        self.lang_encoder = nn.Linear(1000, embedding_size)

    def forward(self, padded_query, query_length, visual_pos,
                visual_neg_intra=None, visual_neg_inter=None):
        v_embedding_neg_intra = None
        v_embedding_neg_inter = None
        B = len(padded_query)

        v_embedding_pos = self.img_encoder(visual_pos)
        if visual_neg_intra is not None:
            v_embedding_neg_intra = self.img_encoder(visual_neg_intra)
        if visual_neg_inter is not None:
            v_embedding_neg_inter = self.img_encoder(visual_neg_inter)

        packed_query = pack_padded_sequence(
            padded_query, query_length, batch_first=True)
        packed_output, _ = self.sentence_encoder(packed_query)
        output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                        total_length=self.max_length)
        # TODO: try max-pooling
        last_output = output[range(B), query_length, :]
        l_embedding = self.lang_encoder(last_output)
        return (l_embedding, v_embedding_pos, v_embedding_neg_intra,
                v_embedding_neg_inter)


class ContextGating(nn.Module):
    """GLU transformation to the incoming data

    Args:
        dimension: size of each input sample

    Credits to @antoine77340
    """

    def __init__(self, dimension):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.bn = nn.BatchNorm1d(dimension)

    def forward(self, x):
        x1 = self.fc(x)
        x1 = self.bn(x1)
        x = torch.cat((x, x1), 1)
        return F.glu(x, 1)


class GatedEmbedding(nn.Module):
    """Non-linear transformation to the incoming data

    Args:
        in_features: size of each input sample
        out_features: size of each output sample

    Credits to @antoine77340
    """

    def __init__(self, in_features, out_features):
        super(GatedEmbedding, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.cg = ContextGating(out_features)

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        x = F.normalize(x)
        return x


class MEE(nn.Module):
    """Mixture of Embedding Experts

    Args:
        video_modality_dim: dict.
        text_dim: size of each output sample.
    TODO:
        weighted sum as einsum. caveat: emmbbedings of different size.

    Credits to @antoine77340
    """

    def __init__(self, text_dim, video_modality_dim):
        super(MEE, self).__init__()
        self.m = list(video_modality_dim.keys())
        video_GU, text_GU = zip(*[(GatedEmbedding(dim[0], dim[1]),
                                   GatedEmbedding(text_dim, dim[1]))
                                  for _, dim in video_modality_dim.items()])
        self.video_GU = nn.ModuleList(video_GU)
        self.text_GU = nn.ModuleList(text_GU)
        self.moe_fc = nn.Linear(text_dim, len(video_modality_dim))

    def forward(self, text, video):
        # Note: available and conf stuff were removed
        B, M = len(text), len(self.m)
        v_embedding, t_embedding = {}, {}
        for i, l in enumerate(self.video_GU):
            v_embedding[self.m[i]] = l(video[self.m[i]])
        for i, l in enumerate(self.text_GU):
            t_embedding[self.m[i]] = l(text)

        # MOE weights computation + normalization
        moe_weights = self.moe_fc(text)
        moe_weights = F.softmax(moe_weights, dim=1)
        norm_weights = torch.sum(moe_weights, dim=1)
        norm_weights = norm_weights.unsqueeze(1)
        moe_weights = torch.div(moe_weights, norm_weights)

        # TODO: search a clean way to do this
        scores_m_list = [None for i in range(len(self.m))]
        for i, m in enumerate(v_embedding):
            w_similarity = (moe_weights[:, i:i + 1] *
                            t_embedding[m] * v_embedding[m])
            scores_m_list[i] = torch.sum(w_similarity, dim=-1, keepdim=True)
        scores_m = torch.cat(scores_m_list, dim=1)
        scores = torch.sum(scores_m, dim=-1)
        return scores


if __name__ == '__main__':
    import torch, random
    from torch.nn.utils.rnn import pad_sequence
    B, LD = 3, 5
    net = MCN(lang_size=LD, rec_layers=1)
    x = torch.rand(B, 4096, requires_grad=True)
    z = [random.randint(2, 6) for i in range(B)]
    z.sort(reverse=True)
    y = [torch.rand(i, LD, requires_grad=True) for i in z]
    y_padded = pad_sequence(y, True)
    a, b, c, d = net(y_padded, z, x, x, x)
    a, b, *c = net(y_padded, z, x)
    b.backward(b.clone())
    # Unsuccesful attempt tp check backward
    # b.backward(10000*b.clone())
    # print(z)
    # print(y_padded)
    # print(f'y.shape = {y_padded.shape}')
    # print(y_padded.grad)
    # print([i.grad for i in y])

    # simple test ContextGating
    d = 50
    net = ContextGating(d)
    x = torch.rand(B, d, requires_grad=True)
    y = net(x)
    y.backward(y.clone())

    # simple test GatedEmbeddingUnit
    d = 50
    net = GatedEmbedding(d, 2*d)
    x = torch.rand(B, d, requires_grad=True)
    y = net(x)
    y.backward(y.clone())

    # simple test MEE
    xd, yd1, yd2 = 15, (10, 20), (14, 14)
    yshape = {'1': yd1, '2': yd2}
    net = MEE(xd, yshape)
    x = torch.rand(B, xd, requires_grad=True)
    y = {i: torch.rand(B, v[0], requires_grad=True)
         for i, v in yshape.items()}
    z = net(x, y)
    z.backward(z.clone())
    # print(z.requires_grad, z.grad_fn)
    # Checking backward (edit forward to expose those variable again)
    # print(net.scores_m.requires_grad, net.scores_m.grad_fn)
    # print([(i.requires_grad, i.grad_fn) for i in net.scores_lst])