import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from chamfer import DoubleMaskedChamferDistance
import copy
MOMENT_RETRIEVAL_MODELS = ['MCN', 'SMCN', 'SMCNTalcv1', 'CALChamfer',
                           'LateFusion']


class MCN(nn.Module):
    """MCN model
    TODO:
        try max pooling
        (pr): compare runtime with LSTM cell
    """

    def __init__(self, visual_size={'rgb':4096}, lang_size=300, embedding_size=100,
                 dropout=0.3, max_length=None, visual_hidden=500,
                 lang_hidden=1000, visual_layers=1, unit_vector=False,
                 bi_lstm=False, lang_dropout=0.0, alpha=0.0):
        super(MCN, self).__init__()
        self.embedding_size = embedding_size
        self.max_length = max_length
        self.unit_vector = unit_vector

        self.lang_hidden = lang_hidden
        self.lang_size = lang_size
        self.bi_lstm = bi_lstm
        self.lang_dropout = lang_dropout
        bi_norm = 1
        if self.bi_lstm:
            bi_norm = 2

        self.keys = list(visual_size.keys())

        self.visual_encoder = nn.ModuleDict({})
        for key in self.keys:
            
            visual_encoder = [nn.Linear(visual_size[key], visual_hidden),
                          nn.ReLU(inplace=True)]
            # (optional) add depth to visual encoder (capacity)
            for i in  range(visual_layers - 1):
                visual_encoder += [nn.Linear(visual_hidden, visual_hidden),
                                nn.ReLU(inplace=True)]
            self.visual_encoder[key] = nn.Sequential(
                *visual_encoder,
                nn.Linear(visual_hidden, embedding_size),
                nn.Dropout(dropout)
            )
        # self.visual_embedding = nn.Sequential(
        #     nn.Linear(visual_hidden, embedding_size),
        #     nn.Dropout(dropout)
        # )

        self.sentence_encoder = nn.LSTM(self.lang_size, self.lang_hidden,
                                batch_first=True, bidirectional=self.bi_lstm)
        self.lang_encoder = nn.Linear(
            bi_norm * self.lang_hidden, self.embedding_size)
        self.init_parameters()

    def forward(self, padded_query, query_length, visual_pos,
                visual_neg_intra=None, visual_neg_inter=None):
        # v_v_embedded_* are tensors of shape [B, D]
        (v_embedded_pos, v_embedded_neg_intra,
         v_embedded_neg_inter) = self.encode_visual(
             visual_pos, visual_neg_intra, visual_neg_inter)

        l_embedded = self.encode_query(padded_query, query_length)
        c_pos = {k:self.compare_emdeddings(l_embedded, 
                                v_embedded_pos[k]) for k in self.keys}
        c_neg_intra, c_neg_inter = None, None
        
        condition_neg_intra = v_embedded_neg_intra[self.keys[0]]
        if condition_neg_intra is not None:
            c_neg_intra = {k:self.compare_emdeddings(l_embedded, 
                            v_embedded_neg_intra[k]) for k in self.keys}
        
        condition_neg_inter = v_embedded_neg_inter[self.keys[0]]
        if condition_neg_inter is not None:
            c_neg_inter = {k:self.compare_emdeddings(l_embedded, 
                            v_embedded_neg_inter[k]) for k in self.keys}
        return c_pos, c_neg_intra, c_neg_inter

    def encode_visual(self, pos, neg_intra, neg_inter):
        pos, neg_intra, neg_inter = self._unpack_visual(
            pos, neg_intra, neg_inter)
        embedded_neg_intra, embedded_neg_inter = None, None

        embedded_pos = {k:self.visual_encoder[k](v) for k,v in pos.items()}
        if self.unit_vector:
            embedded_pos = {k:F.normalize(embedded_pos[k], dim=-1) for k in self.keys}
        condition_neg_intra = neg_intra[self.keys[0]]
        if condition_neg_intra is not None:
            embedded_neg_intra = {k:self.visual_encoder[k](neg_intra[k]) for k in self.keys}
            if self.unit_vector:
                embedded_neg_intra = {k:F.normalize(embedded_neg_intra[k], dim=-1) for k in self.keys}
                                    
        condition_neg_inter = neg_inter[self.keys[0]]
        if condition_neg_inter is not None:
            embedded_neg_inter = {k:self.visual_encoder[k](neg_inter[k]) for k in self.keys}                     
            if self.unit_vector:
                embedded_neg_inter = {k:F.normalize(embedded_neg_inter[k], dim=-1) for k in self.keys}  
                                    
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
        embedded_lang = self.lang_encoder(last_output)
        if self.unit_vector:
            embedded_lang = F.normalize(embedded_lang, dim=-1)
        return embedded_lang

    def compare_emdeddings(self, anchor, x, dim=-1):
        # TODO: generalize to other similarities
        # cos_sim = F.cosine_similarity(anchor,x,dim=dim)
        # return 1 - cos_sim
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

    def optimization_parameters(
            self, initial_lr=1e-2, caffe_setup=False, freeze_visual=False,
            freeze_lang=False):
        # freeze_visual_encoder=True):
        if caffe_setup:
            return self.optimization_parameters_original(
                initial_lr, freeze_visual, freeze_lang)
                # freeze_visual_encoder)
        prm_policy = [
            {'params': self.sentence_encoder.parameters(),
             'lr': initial_lr * 10},
            {'params': self.visual_encoder.parameters()},
            {'params': self.lang_encoder.parameters()},
        ]
        return prm_policy

    def optimization_parameters_original(
            self, initial_lr, freeze_visual, freeze_lang):
        # freeze_visual_encoder):
        prm_policy = []

        for name, prm in self.named_parameters():
            is_lang_tower = ('sentence_encoder' in name or
                             'lang_encoder' in name)
            is_visual_tower = 'visual_encoder' in name
            # is_visual_tower = ('visual_encoder' in name or
            #                    'visual_embedding' in name)
            if freeze_visual and is_visual_tower:
                continue
            elif freeze_lang and is_lang_tower:
                continue
            # elif freeze_visual_encoder and 'visual_encoder' in name:
            #     continue
            elif 'sentence_encoder' in name and 'bias_ih_l' in name:
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
        d_pos = torch.stack([v for _,v in d_pos.items()]).sum(dim=0)
        return d_pos, False

    def search(self, query, table):
        "Exhaustive search of single query in table"
        return self.compare_emdeddings(query, table), False

    def _unpack_visual(self, *args):
        "Get visual feature inside a dict"
        argout = []
        for i in args:
            if isinstance(i, dict):
                argout.append(i)
            else:
                argout.append(i)
        return argout


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
        pos, neg_intra, neg_inter = self._unpack_visual(pos, neg_intra, neg_inter)
        embedded_neg_intra, embedded_neg_inter = None, None

        embedded_pos = {k:self.fwd_visual_snippets(k,pos[k]) for k in self.keys}
                                
        if neg_intra['mask'] is not None:
            embedded_neg_intra = {k:self.fwd_visual_snippets(k,neg_intra[k]) for k in self.keys}
                                    
        if neg_inter['mask'] is not None:
            embedded_neg_inter = {k:self.fwd_visual_snippets(k,neg_inter[k]) for k in self.keys}
                                    
        return embedded_pos, embedded_neg_intra, embedded_neg_inter

    def fwd_visual_snippets(self, key, x):
        B, N, D = x.shape
        x_ = x.view(-1, D)
        x_ = self.visual_encoder[key](x_)
        if self.unit_vector:
            x_ = F.normalize(x_)
        return x_.view((B, N, -1))

    def pool_compared_snippets(self, x, mask):
        masked_x = x * mask
        K = mask.detach().sum(dim=-1)
        return masked_x.sum(dim=-1) / K

    def compare_emdedded_snippets(self, embedded_p, embedded_n_intra,
                                  embedded_n_inter, embedded_a,
                                  pos, neg_intra, neg_inter):
        pos, neg_intra, neg_inter = self._unpack_visual(pos, neg_intra, neg_inter)

        mask_p, mask_n_intra, mask_n_inter = pos['mask'], neg_intra['mask'], neg_inter['mask']
        c_neg_intra, c_neg_inter = None, None

        c_pos = {k:self.pool_compared_snippets(
                self.compare_emdeddings(embedded_a, embedded_p[k]), mask_p) for k in self.keys}
                
        if mask_n_intra is not None:
            c_neg_intra = {k:self.pool_compared_snippets(
                self.compare_emdeddings(embedded_a, embedded_n_intra[k]),mask_n_intra) for k in self.keys}
            
        if mask_n_inter is not None:
            c_neg_inter = {k:self.pool_compared_snippets(
                self.compare_emdeddings(embedded_a, embedded_n_inter[k]),mask_n_inter) for k in self.keys}
        return c_pos, c_neg_intra, c_neg_inter

    def search(self, query, table, clips_per_segment, clips_per_segment_list):
        """Exhaustive search of query in table

        TODO: batch to avoid out of memory?
        """
        clip_distance = self.compare_emdeddings(
            query, table).split(clips_per_segment_list)
        sorted_clips_per_segment, ind = clips_per_segment.sort(
            descending=True)
        # distance >= 0 thus we pad with zeros
        clip_distance_padded = pad_sequence(
            [clip_distance[i] for i in ind], batch_first=True)
        sorted_segment_distance = (clip_distance_padded.sum(dim=1) /
                                   sorted_clips_per_segment)
        _, original_ind = ind.sort(descending=False)
        segment_distance = sorted_segment_distance[original_ind]
        return segment_distance, False

    def _unpack_visual(self, *args):
        "Get visual feature inside a dict"
        argout = []
        keys = args[0].keys()
        for i in args:
            if isinstance(i, dict):
                # assert len(i) == 2 or len(i)==3
                # only works in cpython >= 3.6
                argout.append(i)
            elif i is None:
                argout.append({k:None for k in keys})
            else:
                argout.append(i)
        return argout


class SMCNCL(SMCN):
    "SMCN model plus single clip score"

    def __init__(self, *args, **kwargs):
        super(SMCNCL, self).__init__(*args, **kwargs)

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
        # sorry this is a mega empanada. Yes, it's deadline time!
        argout = self.compare_emdedded_snippets(
            v_embedded_pos, v_embedded_neg_intra, v_embedded_neg_inter,
            l_embedded, visual_pos, visual_neg_intra, visual_neg_inter)
        return argout

    def compare_emdedded_snippets(self, embedded_p, embedded_n_intra,
                                  embedded_n_inter, embedded_a,
                                  pos, neg_intra, neg_inter):
        all_masks = self._get_masks(pos, neg_intra, neg_inter)
        c_neg_intra, c_neg_inter = None, None
        c_neg_intra_clip, c_neg_inter_clip = None, None

        mask_p, mask_p_clip = all_masks[:2]
        c_pos = self.pool_compared_snippets(
            self.compare_emdeddings(embedded_a, embedded_p), mask_p)
        c_pos_clip = self.pool_compared_snippets(
            self.compare_emdeddings(embedded_a, embedded_p), mask_p_clip)
        if embedded_n_intra is not None:
            mask_n_intra, mask_n_intra_clip = all_masks[2:4]
            c_neg_intra = self.pool_compared_snippets(
                self.compare_emdeddings(embedded_a, embedded_n_intra),
                mask_n_intra)
            c_neg_intra_clip = self.pool_compared_snippets(
                self.compare_emdeddings(embedded_a, embedded_n_intra),
                mask_n_intra_clip)
        if embedded_n_inter is not None:
            mask_n_inter, mask_n_inter_clip = all_masks[4:]
            c_neg_inter = self.pool_compared_snippets(
                self.compare_emdeddings(embedded_a, embedded_n_inter),
                mask_n_inter)
            c_neg_inter_clip = self.pool_compared_snippets(
                self.compare_emdeddings(embedded_a, embedded_n_inter),
                mask_n_inter_clip)
        return (c_pos, c_neg_intra, c_neg_inter,
                c_pos_clip, c_neg_intra_clip, c_neg_inter_clip)

    def _unpack_visual(self, *args):
        """Get visual feature and mask inside a dict

        You must add the keys into the dict such that you respect the order
        feat, mask, ...

        Note:
            - Assumes cpython >= 3.6
        """
        argout = ()
        for i in args:
            if isinstance(i, dict):
                # only works in cpython >= 3.6
                # TODO: hotspot, cuando se parten los pistones.
                argout += tuple(i.values())[:2]
            elif i is None:
                argout += (None, None)
            else:
                argout += (i,)
        return argout

    def _get_masks(self, *args):
        "Get masks inside a dict. similar to `_unpack_visual`"
        argout = ()
        for i in args:
            if isinstance(i, dict):
                # only works in
                argout += tuple(i.values())[1:]
            elif i is None:
                argout += (None, None)
            else:
                argout += (i,)
        return argout


class SMCNTalcv1(SMCN):
    "SMCNTalc model version 1"

    def __init__(self, num_chunks=6, *args, **kwargs):
        self.num_chunks = num_chunks
        super(SMCNTalcv1, self).__init__(*args, **kwargs)
        # TODO: add capacity to this caga
        self.lang_res_branch = nn.Linear(self.lang_encoder.in_features,
                                         num_chunks * self.embedding_size)

    def forward(self, padded_query, query_length, visual_pos,
                visual_neg_intra=None, visual_neg_inter=None):
        # v_v_embedded_* are tensors of shape [B, N, D]
        (v_embedded_pos, v_embedded_neg_intra,
         v_embedded_neg_inter) = self.encode_visual(
             visual_pos, visual_neg_intra, visual_neg_inter)

        l_embedded = self.encode_query(padded_query, query_length)

        # meta-comparison
        c_pos, c_neg_intra, c_neg_inter = self.compare_emdedded_snippets(
            v_embedded_pos, v_embedded_neg_intra, v_embedded_neg_inter,
            l_embedded, visual_pos, visual_neg_intra, visual_neg_inter)
        return c_pos, c_neg_intra, c_neg_inter

    def encode_query(self, padded_query, query_length):
        B = len(padded_query)
        packed_query = pack_padded_sequence(
            padded_query, query_length, batch_first=True)
        packed_output, _ = self.sentence_encoder(packed_query)
        output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                        total_length=self.max_length)
        # TODO: try max-pooling
        last_output = output[range(B), query_length - 1, :]
        # 1st alternative
        residual = self.lang_encoder(last_output)
        extra_stuff = self.lang_res_branch(last_output)
        # MLP generates time... gimme that sh!t to get an extra day XD
        # create time dimension [B, S*ES] -> [B, S, ES]
        extra_stuff = extra_stuff.view(B, -1, self.embedding_size)
        # create time dimension [B, ES] -> [B, 1, ES]
        residual = residual.view(B, 1, -1)
        language_code = residual + extra_stuff
        return language_code

    def search(self, query, table):
        "Exhaustive search of query in table"
        raise NotImplementedError


class CALChamfer(SMCN):
    '''CALChamfer

    Extends SMCN by adding the chamfer distance as metric for computing the
    distance between two sets of embeddings. The description is encoded
    through an LSTM layer of which every hidden state is mapped to the
    embedding state and constitute the set of language embeddings.
    '''

    def __init__(self, *args, **kwargs):
        super(CALChamfer, self).__init__(*args, **kwargs)
        bi_norm = 1
        if self.bi_lstm:
            bi_norm = 2
        self.iter = 0

        # Language branch
        del self.lang_encoder
        self.sentence_encoder = nn.LSTM(
            self.lang_size, self.lang_hidden, batch_first=True,
            bidirectional=self.bi_lstm, num_layers=1)
        self.state_encoder = nn.Linear(
            bi_norm * self.lang_hidden, self.embedding_size)
        self.chamfer_distance = DoubleMaskedChamferDistance()
        self.init_parameters()

    def forward(self, padded_query, query_length, visual_pos,
                visual_neg_intra=None, visual_neg_inter=None):
        # v_v_embedded_* are tensors of shape [B, N, D]
        (v_embedded_pos, v_embedded_neg_intra,
         v_embedded_neg_inter) = self.encode_visual(
             visual_pos, visual_neg_intra, visual_neg_inter)

        l_embedded = self.encode_query(padded_query, query_length)
        l_mask = self._gen_lan_mask(self.max_length,query_length)

        #embedding normalization
        if self.unit_vector:
            l_embedded = F.normalize(l_embedded, dim=-1)
            v_embedded_pos = F.normalize(v_embedded_pos, dim=-1)
            if v_embedded_neg_intra is not None:
                v_embedded_neg_intra = F.normalize(v_embedded_neg_intra, dim=-1)
            if v_embedded_neg_inter is not None:
                v_embedded_neg_inter = F.normalize(v_embedded_neg_inter, dim=-1)
        else:
            if self.lang_dropout > 0:
                l_embedded = l_embedded*(1-self.lang_dropout)

        # meta-comparison
        c_pos, c_neg_intra, c_neg_inter = self.compare_emdedded_snippets(
            v_embedded_pos, v_embedded_neg_intra, v_embedded_neg_inter,
            l_embedded, l_mask, visual_pos, visual_neg_intra, visual_neg_inter)
        return c_pos, c_neg_intra, c_neg_inter

    def encode_query(self, padded_query, query_length):
        B = len(padded_query)
        packed_query = pack_padded_sequence(padded_query, query_length,
                                            batch_first=True)
        packed_output, _ = self.sentence_encoder(packed_query)
        output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                        total_length=self.max_length)

        # Pass hidden states though a shared linear layer.
        embedded_lang = self.state_encoder(output)

        #Apply dropout if set.
        if self.lang_dropout > 0:
            embedded_lang = self._apply_dopout_language(embedded_lang)

        return embedded_lang

    def encode_visual(self, pos, neg_intra, neg_inter):
        pos, _, neg_intra, _, neg_inter, _ = \
                self._unpack_visual(pos, neg_intra, neg_inter)
        embedded_neg_intra, embedded_neg_inter = None, None

        embedded_pos = self.fwd_visual_snippets(pos)
        if neg_intra is not None:
            embedded_neg_intra = self.fwd_visual_snippets(neg_intra)

        if neg_inter is not None:
            embedded_neg_inter = self.fwd_visual_snippets(neg_inter)

        return embedded_pos, embedded_neg_intra, embedded_neg_inter

    def compare_emdedded_snippets(self, embedded_p, embedded_n_intra,
                                  embedded_n_inter, embedded_a, l_mask,
                                  pos, neg_intra, neg_inter):
        _, mask_p,_, mask_n_intra, _, mask_n_inter = \
            self._unpack_visual(pos, neg_intra, neg_inter)

        c_neg_intra, c_neg_inter = None, None

        c_pos = self.compare_emdeddings(embedded_p, embedded_a, mask_p, l_mask)
        if embedded_n_intra is not None:
            c_neg_intra = self.compare_emdeddings(embedded_n_intra, embedded_a,
                                                    mask_n_intra, l_mask)
        if embedded_n_inter is not None:
            c_neg_inter = self.compare_emdeddings(embedded_n_inter, embedded_a,
                                                    mask_n_inter, l_mask)
        return c_pos, c_neg_intra, c_neg_inter

    def compare_emdeddings(self, v, l, mv, ml):
        return self.chamfer_distance(v, l, mv, ml)

    def predict(self, *args):
        "Compute distance between visual and sentence"
        d_pos, *_ = self.forward(*args)
        return d_pos, False

    def _apply_dopout_language(self, emb_l):
        B = emb_l.size()[0]
        emb_l = emb_l.unsqueeze(1).view(B, -1, self.embedding_size)
        emb_l = emb_l.unsqueeze(dim=2)
        emb_l = F.dropout2d(emb_l,p=self.lang_dropout, training=self.training)
        emb_l = emb_l.squeeze(dim=2)
        return emb_l

    def _gen_lan_mask(self,num_words,query_length):
        #TO DO: move this task to dataset_untrimmed
        B = query_length.size()[0]              # Batch size
        mask = torch.zeros(B,num_words)         # Mask initialization to zero
        # mask fill in with lenght of each single query
        for i in range(B):
            mask[i,0:query_length[i]] = 1
        return mask

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


class old_SMCN(MCN):
    "SMCN model"

    def __init__(self, *args, **kwargs):
        super(old_SMCN, self).__init__(*args, **kwargs)
        visual_size = kwargs['visual_size']['rgb']
        visual_hidden = kwargs['visual_hidden']
        visual_layers = kwargs['visual_layers']
        embedding_size = kwargs['embedding_size']
        dropout = kwargs['dropout']
        visual_encoder = [nn.Linear(visual_size, visual_hidden),
                          nn.ReLU(inplace=True)]
        # (optional) add depth to visual encoder (capacity)
        for i in  range(visual_layers - 1):
            visual_encoder += [nn.Linear(visual_hidden, visual_hidden),
                               nn.ReLU(inplace=True)]
        self.visual_encoder = nn.Sequential(
            *visual_encoder,
            nn.Linear(visual_hidden, embedding_size),
            nn.Dropout(dropout)
        )

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
        pos, neg_intra, neg_inter = self._unpack_visual(pos, neg_intra, neg_inter)
        embedded_neg_intra, embedded_neg_inter = None, None

        embedded_pos = {k:self.fwd_visual_snippets(k,v) 
                                for k,v in pos.items() if k != 'mask'}
        if neg_intra['mask'] is not None:
            embedded_neg_intra = {k:self.fwd_visual_snippets(k,v) 
                                    for k,v in neg_intra.items() if k != 'mask'}   
        if neg_inter['mask'] is not None:
            embedded_neg_inter = {k:self.fwd_visual_snippets(k,v) 
                                    for k,v in neg_inter.items() if k != 'mask'}
        return embedded_pos, embedded_neg_intra, embedded_neg_inter

    def fwd_visual_snippets(self, key, x):
        B, N, D = x.shape
        x_ = x.view(-1, D)
        x_ = self.visual_encoder[key](x_)
        if self.unit_vector:
            x_ = F.normalize(x_)
        return x_.view((B, N, -1))

    def pool_compared_snippets(self, x, mask):
        masked_x = x * mask
        K = mask.detach().sum(dim=-1)
        return masked_x.sum(dim=-1) / K

    def compare_emdedded_snippets(self, embedded_p, embedded_n_intra,
                                  embedded_n_inter, embedded_a,
                                  pos, neg_intra, neg_inter):
        pos, neg_intra, neg_inter = self._unpack_visual(pos, neg_intra, neg_inter)

        mask_p, mask_n_intra, mask_n_inter = pos['mask'], neg_intra['mask'], neg_inter['mask']
        c_neg_intra, c_neg_inter = None, None

        c_pos = {k:self.pool_compared_snippets(
                self.compare_emdeddings(embedded_a, v), mask_p) for k,v in embedded_p.items()}
        if mask_n_intra is not None:
            c_neg_intra = {k:self.pool_compared_snippets(
                self.compare_emdeddings(embedded_a, v),mask_n_intra) for k,v in embedded_n_intra.items()}
        if mask_n_inter is not None:
            c_neg_inter = {k:self.pool_compared_snippets(
                self.compare_emdeddings(embedded_a, v),mask_n_inter) for k,v in embedded_n_inter.items()}
        return c_pos, c_neg_intra, c_neg_inter

    def search(self, query, table, clips_per_segment, clips_per_segment_list):
        """Exhaustive search of query in table
        TODO: batch to avoid out of memory?
        """
        clip_distance = self.compare_emdeddings(
            query, table).split(clips_per_segment_list)
        sorted_clips_per_segment, ind = clips_per_segment.sort(
            descending=True)
        # distance >= 0 thus we pad with zeros
        clip_distance_padded = pad_sequence(
            [clip_distance[i] for i in ind], batch_first=True)
        sorted_segment_distance = (clip_distance_padded.sum(dim=1) /
                                   sorted_clips_per_segment)
        _, original_ind = ind.sort(descending=False)
        segment_distance = sorted_segment_distance[original_ind]
        return segment_distance, False

    def _unpack_visual(self, *args):
        "Get visual feature inside a dict"
        argout = []
        keys = args[0].keys()
        for i in args:
            if isinstance(i, dict):
                # assert len(i) == 2 or len(i)==3
                # only works in cpython >= 3.6
                argout.append(i)
            elif i is None:
                argout.append({k:None for k in keys})
            else:
                argout.append(i)
        return argout


class LateFusion(SMCN):
    "Commbine SMCN with resnet features and  SMCN with obj features"
    def __init__(self, **arch_setup):
        super(LateFusion, self).__init__()
        # Instantiate models
        arch_resnet = {k:v for k,v in arch_setup.items() if k != 'visual_size' or k!='alpha'}
        arch_resnet['visual_size'] = {'rgb': arch_setup['visual_size']['rgb']}
        self.resnet = SMCN(**arch_resnet)

        arch_obj_feat = {k:v for k,v in arch_setup.items() if k != 'visual_size' or k!='alpha'}
        arch_obj_feat['visual_size'] = {'obj': arch_setup['visual_size']['obj']}
        self.obj_feat = SMCN(**arch_obj_feat)
        self.alpha = arch_setup['alpha']

    def predict(self, *args):
        "Compute scores for both models"
        # pass data as it is to the models
        d1, *_ = self.resnet.forward(*args)
        d2, *_ = self.obj_feat.forward(*args)
        # combine the scores
        d_pos =  self.alpha * d1['rgb'] + (1-self.alpha) * d2['obj']
        return d_pos, False

    def load_state_dict(self,snapshots):
        self.resnet.load_state_dict(snapshots)
        snap = './data/interim/didemo/best_models/smcn_tef_weighted_avg_glove_42.pth.tar'
        snapshots = torch.load(snap).get('state_dict')
        self.obj_feat.load_state_dict(snapshots)

    def eval(self):
        self.resnet.eval()
        self.obj_feat.eval()


if __name__ == '__main__':
    import torch, random
    from torch.nn.utils.rnn import pad_sequence
    B, LD = 3, 5
    net = MCN(lang_size=LD)
    x = torch.rand(B, 4096, requires_grad=True)
    z = [random.randint(2, 6) for i in range(B)]
    z.sort(reverse=True)
    y = [torch.rand(i, LD, requires_grad=True) for i in z]
    y_padded = pad_sequence(y, True)
    z = torch.tensor(z)
    a, b, c = net(y_padded, z, x, x, x)
    b.backward(b.clone())
    a, b, *c = net(y_padded, z, x)
    # Unsuccesful attempt tp check backward
    # b.backward(10000*b.clone())
    # print(z)
    # print(y_padded)
    # print(f'y.shape = {y_padded.shape}')
    # print(y_padded.grad)
    # print([i.grad for i in y])

    # SMCNTalcv1
    B, LD, S = 3, 5, 4
    net = SMCNTalcv1(num_chunks=S, lang_size=LD)
    x = torch.rand(B, S, 4096, requires_grad=True)
    m = []
    for i in range(B):
        m_i = torch.zeros(S)
        m_i[:random.randint(0, S) + 1] = 1
        m.append(m_i)
    m = torch.stack(m, 0).requires_grad_()
    x_packed = {'hola': x, 'mask': m}
    z = [random.randint(2, 6) for i in range(B)]
    z.sort(reverse=True)
    y = [torch.rand(i, LD, requires_grad=True) for i in z]
    y_padded = pad_sequence(y, True)
    z = torch.tensor(z)
    a, b, c = net(y_padded, z, x_packed, x_packed, x_packed)
    b.backward(b.clone())
    a, *_ = net(y_padded, z, x_packed)