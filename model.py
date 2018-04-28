import torch.nn as nn
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
                visual_neg_intra, visual_neg_inter):
        B = len(padded_query)
        v_embedding_pos = self.img_encoder(visual_pos)
        v_embedding_neg_intra = self.img_encoder(visual_neg_intra)
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
    # Unsuccesful attempt tp check backward
    # b.backward(10000*b.clone())
    # print(z)
    # print(y_padded)
    # print(f'y.shape = {y_padded.shape}')
    # print(y_padded.grad)
    # print([i.grad for i in y])