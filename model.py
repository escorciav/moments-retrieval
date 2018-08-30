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
                 dropout=0.3, max_length=None, visual_hidden=500,
                 lang_hidden=1000, visual_layers=1):
        super(MCN, self).__init__()
        self.embedding_size = embedding_size
        self.max_length = max_length

        visual_encoder = [nn.Linear(visual_size, visual_hidden),
                          nn.ReLU(inplace=True)]
        # (optional) add depth to visual enconder (capacity)
        for i in  range(visual_layers - 1):
            visual_encoder += [nn.Linear(visual_hidden, visual_hidden),
                               nn.ReLU(inplace=True)]
        self.visual_encoder = nn.Sequential(
          *visual_encoder, nn.Linear(visual_hidden, embedding_size),
          nn.Dropout(dropout)
        )

        self.sentence_encoder = nn.LSTM(
            lang_size, lang_hidden, batch_first=True)
        self.lang_encoder = nn.Linear(lang_hidden, embedding_size)
        self.init_parameters()

    def forward(self, padded_query, query_length, visual_pos,
                visual_neg_intra=None, visual_neg_inter=None):
        # v_v_embedded_* are tensors of shape [B, D]
        (v_embedded_pos, v_embedded_neg_intra,
         v_embedded_neg_inter) = self.encode_visual(
             visual_pos, visual_neg_intra, visual_neg_inter)

        l_embedded = self.encode_query(padded_query, query_length)

        c_pos = self.compare_emdeddings(l_embedded, v_embedded_pos)
        c_neg_intra, c_neg_inter = None, None
        if v_embedded_neg_intra is not None:
            c_neg_intra = self.compare_emdeddings(
                l_embedded, v_embedded_neg_intra)
        if v_embedded_neg_inter is not None:
            c_neg_inter = self.compare_emdeddings(
                l_embedded, v_embedded_neg_inter)
        return c_pos, c_neg_intra, c_neg_inter

    def encode_visual(self, pos, neg_intra, neg_inter):
        pos, neg_intra, neg_inter = self._unpack_visual(
            pos, neg_intra, neg_inter)
        embedded_neg_intra, embedded_neg_inter = None, None

        embedded_pos = self.visual_encoder(pos)
        if neg_intra is not None:
            embedded_neg_intra = self.visual_encoder(neg_intra)
        if neg_inter is not None:
            embedded_neg_inter = self.visual_encoder(neg_inter)
        return embedded_pos, embedded_neg_intra, embedded_neg_inter

    def encode_query(self, padded_query, query_length):
        B = len(padded_query)
        packed_query = pack_padded_sequence(
            padded_query, query_length, batch_first=True)
        packed_output, _ = self.sentence_encoder(packed_query)
        output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                        total_length=self.max_length)
        # TODO: try max-pooling
        last_output = output[range(B), query_length - 1, :]
        return self.lang_encoder(last_output)

    def compare_emdeddings(self, anchor, x, dim=-1):
        # TODO: generalize to other similarities
        return (anchor - x).pow(2).sum(dim=dim)

    def init_parameters(self):
        "Initialize network parameters"
        # if filename is not None and os.path.exists(filename):
        #    raise NotImplementedError('WIP')
        for name, prm in self.named_parameters():
            if 'bias' in name:
                prm.data.fill_(0)
            else:
                prm.data.uniform_(-0.08, 0.08)

    def optimization_parameters(self, initial_lr=1e-2, caffe_setup=False):
        if caffe_setup:
            return self.optimization_parameters_original(initial_lr)
        prm_policy = [
            {'params': self.sentence_encoder.parameters(),
             'lr': initial_lr * 10},
            {'params': self.visual_encoder.parameters()},
            {'params': self.lang_encoder.parameters()},
        ]
        return prm_policy

    def optimization_parameters_original(self, initial_lr):
        prm_policy = []
        for name, prm in self.named_parameters():
            if 'sentence_encoder' in name and 'bias_ih_l' in name:
                continue
            elif 'sentence_encoder' in name:
                prm_policy.append({'params': prm, 'lr': initial_lr * 10})
            elif 'bias' in name:
                prm_policy.append({'params': prm, 'lr': initial_lr * 2})
            else:
                prm_policy.append({'params': prm})
        return prm_policy

    def predict(self, *args):
        "Compute distance between visual and sentence"
        d_pos, *_ = self.forward(*args)
        return d_pos, False

    def _unpack_visual(self, *args):
        "Get visual feature inside a dict"
        argout = ()
        for i in args:
            if isinstance(i, dict):
                assert len(i) == 1
                j = next(iter(i))
                argout += (i[j],)
            else:
                argout += (i,)
        return argout


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


class TripletMEE(nn.Module):
    """MEE model with triplets
    TODO:
        Improve abstraction
    """

    def __init__(self, text_embedding, visual_embedding, word_size=300,
                 lstm_layers=1, max_length=50):
        super(TripletMEE, self).__init__()
        # Setup Text Encoder
        self.max_length = max_length
        self.sentence_encoder = nn.LSTM(
            word_size, text_embedding, lstm_layers, batch_first=True)
        # self.lang_encoder = nn.Linear(1000, embedding_size)
        # Setup MEE
        self.mee = MEE(text_embedding, visual_embedding)

    def forward(self, padded_query, query_length, visual_pos,
                visual_neg_intra=None, visual_neg_inter=None):
        similarity_neg_intra = None
        similarity_neg_inter = None
        B = len(padded_query)

        # Encode sentence
        packed_query = pack_padded_sequence(
            padded_query, query_length, batch_first=True)
        packed_output, _ = self.sentence_encoder(packed_query)
        output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                        total_length=self.max_length)
        text_vector = output[range(B), query_length, :]

        # MEE
        similarity_pos = self.mee(text_vector, visual_pos)
        if visual_neg_intra is not None:
            similarity_neg_intra = self.mee(text_vector, visual_neg_intra)
        if visual_neg_inter is not None:
            similarity_neg_inter = self.mee(text_vector, visual_neg_inter)

        return (similarity_pos, similarity_neg_intra,
                similarity_neg_inter)

    def predict(self, *args):
        "Compute similarity between visual and sentence"
        similarity, *_ = self.forward(*args)
        return similarity, True


class SMCN(MCN):
    "SMCN model"

    def __init__(self, *args, **kwargs):
        super(SMCN, self).__init__(*args, **kwargs)

    def forward(self, padded_query, query_length, visual_pos,
                visual_neg_intra=None, visual_neg_inter=None):
        # v_v_embedded_* are tensors of shape [B, N, D]
        (v_embedded_pos, v_embedded_neg_intra,
         v_embedded_neg_inter) = self.encode_visual(
             visual_pos, visual_neg_intra, visual_neg_inter)

        l_embedded = self.encode_query(padded_query, query_length)
        # transform l_emdedded into a tensor of shape [B, 1, D]
        l_embedded = l_embedded.unsqueeze(1)

        # meta-comparison
        c_pos, c_neg_intra, c_neg_inter = self.compare_emdedded_snippets(
            v_embedded_pos, v_embedded_neg_intra, v_embedded_neg_inter,
            l_embedded, visual_pos, visual_neg_intra, visual_neg_inter)
        return c_pos, c_neg_intra, c_neg_inter

    def encode_visual(self, pos, neg_intra, neg_inter):
        pos, _, neg_intra, _, neg_inter, _ = self._unpack_visual(
            pos, neg_intra, neg_inter)
        embedded_neg_intra, embedded_neg_inter = None, None

        embedded_pos = self.fwd_visual_snippets(pos)
        if neg_intra is not None:
            embedded_neg_intra = self.fwd_visual_snippets(neg_intra)
        if neg_inter is not None:
            embedded_neg_inter = self.fwd_visual_snippets(neg_inter)
        return embedded_pos, embedded_neg_intra, embedded_neg_inter

    def fwd_visual_snippets(self, x):
        B, N, D = x.shape
        x_ = x.view(-1, D)
        return self.visual_encoder(x_).view((B, N, -1))

    def pool_compared_snippets(self, x, mask):
        masked_x = x * mask
        K = mask.detach().sum(dim=-1)
        return masked_x.sum(dim=-1) / K

    def compare_emdedded_snippets(self, embedded_p, embedded_n_intra,
                                  embedded_n_inter, embedded_a,
                                  pos, neg_intra, neg_inter):
        _, mask_p, _, mask_n_intra, _, mask_n_inter = self._unpack_visual(
            pos, neg_intra, neg_inter)
        c_neg_intra, c_neg_inter = None, None

        c_pos = self.pool_compared_snippets(
            self.compare_emdeddings(embedded_a, embedded_p), mask_p)
        if embedded_n_intra is not None:
            c_neg_intra = self.pool_compared_snippets(
                self.compare_emdeddings(embedded_a, embedded_n_intra),
                mask_n_intra)
        if embedded_n_inter is not None:
            c_neg_inter = self.pool_compared_snippets(
                self.compare_emdeddings(embedded_a, embedded_n_inter),
                mask_n_inter)
        return c_pos, c_neg_intra, c_neg_inter

    def _unpack_visual(self, *args):
        "Get visual feature inside a dict"
        argout = ()
        for i in args:
            if isinstance(i, dict):
                assert len(i) == 2
                # only works in cpython >= 3.6
                argout += tuple(i.values())
            elif i is None:
                argout += (None, None)
            else:
                argout += (i,)
        return argout


if __name__ == '__main__':
    import torch, random
    from torch.nn.utils.rnn import pad_sequence
    B, LD = 3, 5
    net = MCN(lang_size=LD, visual_layers=2)
    x = torch.rand(B, 4096, requires_grad=True)
    z = [random.randint(2, 6) for i in range(B)]
    z.sort(reverse=True)
    y = [torch.rand(i, LD, requires_grad=True) for i in z]
    y_padded = pad_sequence(y, True)
    z = torch.tensor(z, requires_grad=True)
    argout = net(y_padded, z, x, x, x)
    assert len(argout) == 3
    assert all([isinstance(i, torch.Tensor) for i in argout])
    argout = net(y_padded, z, x)
    assert len(argout) == 3
    argout[0].backward(argout[0].clone())
    assert all([i is None for i in argout[1:]])
    print(net)
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

    # TripletMEE
    B = 3
    xd, yd1, yd2 = 15, (10, 20), (14, 14)
    yshape = {'1': yd1, '2': yd2}
    extra = {'word_size': 5, 'lstm_layers':1, 'max_length':11}
    net = TripletMEE(xd, yshape, **extra)
    l = [random.randint(2, extra['max_length']) for i in range(B)]
    l.sort(reverse=True)
    x = [torch.rand(i, extra['word_size'], requires_grad=True) for i in l]
    x_padded = pad_sequence(x, True)
    y = {i: torch.rand(B, v[0], requires_grad=True)
         for i, v in yshape.items()}
    yn = {i: torch.rand(B, v[0], requires_grad=True)
          for i, v in yshape.items()}
    z = net(x_padded, l, y, yn)
    z_ = z[0] + z[1]
    z_.backward(z_.clone())
    assert z[-1] is None