import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from chamfer import DoubleMaskedChamferDistance
import copy
import numpy as np
MOMENT_RETRIEVAL_MODELS = ['MCN', 'SMCN', 'SMCNTalcv1', 'CALChamfer','LateFusion','EarlyFusion']


class MCN(nn.Module):
    """MCN model
    TODO:
        try max pooling
        (pr): compare runtime with LSTM cell
    """

    def __init__(self, visual_size={'rgb':4096}, lang_size=300, embedding_size=100,
                 dropout=0.3, max_length=None, visual_hidden=500,
                 lang_hidden=1000, visual_layers=1, lang_layers=2, unit_vector=False,
                 bi_lstm=False, lang_dropout=0.0, alpha=0.0):
        super(MCN, self).__init__()
        self.embedding_size = embedding_size
        self.max_length = max_length
        self.unit_vector = unit_vector
        self.visual_layers = visual_layers
        self.lang_layers = lang_layers
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
            for _ in  range(self.visual_layers - 1):
                visual_encoder += [nn.Linear(visual_hidden, visual_hidden),
                                nn.ReLU(inplace=True)]
            self.visual_encoder[key] = nn.Sequential(
                *visual_encoder,
                nn.Linear(visual_hidden, embedding_size),
                nn.Dropout(dropout)
            )

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
        
        # condition_neg_intra = v_embedded_neg_intra[self.keys[0]]
        if v_embedded_neg_intra is not None:
            c_neg_intra = {k:self.compare_emdeddings(l_embedded, 
                            v_embedded_neg_intra[k]) for k in self.keys}
        
        # condition_neg_inter = v_embedded_neg_inter[self.keys[0]]
        if v_embedded_neg_inter is not None:
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

        if neg_intra is not None:
            embedded_neg_intra = {k:self.visual_encoder[k](neg_intra[k]) for k in self.keys}
            if self.unit_vector:
                embedded_neg_intra = {k:F.normalize(embedded_neg_intra[k], dim=-1) for k in self.keys}
                                    
        if neg_inter is not None:
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
        y = anchor - x
        return (y * y).sum(dim=dim)

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

    Chamfer distance is applied to all brances Lang-rgb / Lang-obj (if presents)
    '''

    def __init__(self, *args, **kwargs):
        super(CALChamfer, self).__init__(*args, **kwargs)
        bi_norm = 1
        if self.bi_lstm:
            bi_norm = 2

        # Language branch
        self.sentence_encoder = nn.ModuleDict({})
        for key in self.keys:
            self.sentence_encoder[key] = nn.LSTM(
                            self.lang_size, self.lang_hidden, 
                            batch_first=True, bidirectional=self.bi_lstm, 
                            num_layers=1)

        self.state_encoder = nn.ModuleDict({})
        for key in self.keys:
            state_encoder = []
            for _ in range(self.lang_layers - 2):
                state_encoder += [nn.Linear(bi_norm * self.lang_hidden, self.lang_hidden),
                                nn.ReLU(inplace=True)]
            state_encoder += [nn.Linear(bi_norm * self.lang_hidden, int(self.lang_hidden/2)),
                                nn.ReLU(inplace=True)]
            self.state_encoder[key] = nn.Sequential(
                *state_encoder,
                nn.Linear(int(self.lang_hidden/2), self.embedding_size),
                nn.Dropout(self.lang_dropout)
            )

        self.chamfer_distance = DoubleMaskedChamferDistance()
        self.init_parameters()

    def forward(self, padded_query, query_length, visual_pos,
                visual_neg_intra=None, visual_neg_inter=None):
        # v_v_embedded_* are tensors of shape [B, N, D]
        (v_embedded_pos, v_embedded_neg_intra,
         v_embedded_neg_inter) = self.encode_visual(
             visual_pos, visual_neg_intra, visual_neg_inter)

        l_embedded = self.encode_query(padded_query, query_length)
        l_mask = self._gen_lan_mask(self.max_length,query_length).to(l_embedded[self.keys[0]].device)

        #embedding normalization
        if self.unit_vector:
            l_embedded = {k:F.normalize(l_embedded[k], dim=-1) for k in self.keys}
            v_embedded_pos = {k:F.normalize(v_embedded_pos[k], dim=-1) for k in self.keys}
            if v_embedded_neg_intra is not None:
                v_embedded_neg_intra = {k:F.normalize(v_embedded_neg_intra[k], dim=-1) for k in self.keys}
            if v_embedded_neg_inter is not None:
                v_embedded_neg_inter = {k:F.normalize(v_embedded_neg_inter[k], dim=-1) for k in self.keys}
                

        # meta-comparison
        c_pos, c_neg_intra, c_neg_inter = self.compare_emdedded_snippets(
            v_embedded_pos, v_embedded_neg_intra, v_embedded_neg_inter,
            l_embedded, l_mask, visual_pos, visual_neg_intra, visual_neg_inter)
        return c_pos, c_neg_intra, c_neg_inter

    def encode_query(self, padded_query, query_length):
        B = len(padded_query)
        packed_query = pack_padded_sequence(padded_query, 
                                            query_length,
                                            batch_first=True)
        packed_output = {k:self.sentence_encoder[k](packed_query) for k in self.keys}
        output = {k:pad_packed_sequence(packed_output[k][0], batch_first=True,
                                        total_length=self.max_length) for k in self.keys}

        # Pass hidden states though a shared linear layer.
        embedded_lang = {k:self.state_encoder[k](output[k][0]) for k in self.keys}

        return embedded_lang

    def encode_visual(self, pos, neg_intra, neg_inter):
        pos, neg_intra, neg_inter = self._unpack_visual(pos, neg_intra, neg_inter)
        embedded_neg_intra, embedded_neg_inter = None, None

        embedded_pos = {k:self.fwd_visual_snippets(k,pos[k]) for k in self.keys}

        mask_key = '-'.join(['mask',self.keys[0]])          
        if neg_intra[mask_key] is not None:
            embedded_neg_intra = {k:self.fwd_visual_snippets(k,neg_intra[k]) for k in self.keys}
                                    
        if neg_inter[mask_key] is not None:
            embedded_neg_inter = {k:self.fwd_visual_snippets(k,neg_inter[k]) for k in self.keys}
                                    
        return embedded_pos, embedded_neg_intra, embedded_neg_inter

    def compare_emdedded_snippets(self, embedded_p, embedded_n_intra,
                                  embedded_n_inter, embedded_a, l_mask,
                                  pos, neg_intra, neg_inter):
        pos, neg_intra, neg_inter = self._unpack_visual(pos, neg_intra, neg_inter)
        c_neg_intra, c_neg_inter = None, None
        mask_p, mask_n_intra, mask_n_inter = {}, {}, {}
        
        for k in self.keys:
            mask_key = '-'.join(['mask',k])
            mask_p[k] = pos[mask_key]
            mask_n_intra[k] = neg_intra[mask_key]
            mask_n_inter[k] = neg_inter[mask_key]
        
        c_pos = {k:self.compare_emdeddings(embedded_p[k], embedded_a[k], 
                                mask_p[k], l_mask) for k in self.keys}

        if mask_n_intra[self.keys[0]] is not None:
            c_neg_intra = {k:self.compare_emdeddings(embedded_n_intra[k], embedded_a[k],
                                            mask_n_intra[k], l_mask) for k in self.keys}
        if mask_n_inter[self.keys[0]] is not None:
            c_neg_inter = {k:self.compare_emdeddings(embedded_n_inter[k], embedded_a[k],
                                            mask_n_inter[k], l_mask) for k in self.keys}
        return c_pos, c_neg_intra, c_neg_inter

    def compare_emdeddings(self, v, l, mv, ml):
        return self.chamfer_distance(v, l, mv, ml)

    def predict(self, *args):
        "Compute distance between visual and sentence"
        d_pos, *_ = self.forward(*args)
        d_pos = torch.stack([v for _,v in d_pos.items()]).sum(dim=0)
        return d_pos, False

    def _gen_lan_mask(self,num_words,query_length):
        #TO DO: move this task to dataset_untrimmed
        B = query_length.size()[0]              # Batch size
        mask = torch.zeros(B,num_words)         # Mask initialization to zero
        # mask fill in with lenght of each single query
        for i in range(B):
            mask[i,0:query_length[i]] = 1
        return mask

    def search(self, query, query_length, moments, v_mask, batch_size):
        """Exhaustive search of query in table

        TODO: batch to avoid out of memory?
        """

        if batch_size == 0:
            B = moments.shape[0]
            _, d1, d2 = query.size()
            query  = query.expand(B, d1, d2)
            l_mask = self._gen_lan_mask(self.max_length,query_length)
            l_mask = l_mask.expand(B, l_mask.size()[1])
            chamfer_distance = self.compare_emdeddings(moments, query, v_mask, l_mask)

            return chamfer_distance, False

        elif batch_size > 0:
            # Sizes
            B = moments.shape[0]
            batch_size = min(batch_size, B)
            _, d1_q, d2_q = query.size()
            
            # Reshape query and creation query mask
            query_ = query.expand(batch_size, d1_q, d2_q)
            l_mask = self._gen_lan_mask(self.max_length,query_length).to('cuda')
            l_mask_= l_mask.expand(batch_size, l_mask.size()[1])
        
            #Batchify chamfer distance calculation
            chamfer_distance = []
            for i in range(B//batch_size):
                chamfer_distance.extend(
                    self.compare_emdeddings(
                        moments[i*batch_size: (i+1)*batch_size].to('cuda'), query_, 
                        v_mask[i*batch_size:  (i+1)*batch_size].to('cuda'), l_mask_).cpu()
                    )
            # Process reminder of batch
            if batch_size < B:
                reminder = B % batch_size
                query_   = query.to('cuda').expand(reminder, d1_q, d2_q)
                l_mask_  = l_mask.to('cuda').expand(reminder, l_mask.size()[1])
                chamfer_distance.extend(
                    self.compare_emdeddings(
                        moments[-reminder:].to('cuda'), query_, 
                        v_mask[-reminder:].to('cuda') , l_mask_).cpu()
                    )

            return torch.FloatTensor(chamfer_distance), False
        else:
            raise('Batch size chamfer incorrect. Use 0 or a positive value.')

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


class EarlyFusion(SMCN):
    '''CALChamfer

    Extends SMCN by adding the chamfer distance as metric for computing the
    distance between two sets of embeddings. The description is encoded
    through an LSTM layer of which every hidden state is mapped to the
    embedding state and constitute the set of language embeddings.

    Chamfer distance is applied to all brances Lang-rgb / Lang-obj (if presents)
    '''

    def __init__(self, *args, **kwargs):
        super(EarlyFusion, self).__init__(*args, **kwargs)
        bi_norm = 1
        if self.bi_lstm:
            bi_norm = 2

        lang_layers = 2

        # Language branch
        self.sentence_encoder = nn.LSTM(
            self.lang_size, self.lang_hidden, 
            batch_first=True, bidirectional=self.bi_lstm, 
            num_layers=1)

        state_encoder = [nn.Linear(bi_norm * self.lang_hidden, int(self.lang_hidden/2)),
                            nn.ReLU(inplace=True)]
        self.state_encoder = nn.Sequential(
                *state_encoder,
                nn.Linear(int(self.lang_hidden/2), self.embedding_size),
                nn.Dropout(self.lang_dropout)
            )

        self.chamfer_distance = DoubleMaskedChamferDistance()
        self.init_parameters()

    def forward(self, padded_query, query_length, visual_pos,
                visual_neg_intra=None, visual_neg_inter=None):
        # v_v_embedded_* are tensors of shape [B, N, D]
        (v_embedded_pos, v_embedded_neg_intra,
         v_embedded_neg_inter) = self.encode_visual(
             visual_pos, visual_neg_intra, visual_neg_inter)

        l_embedded = self.encode_query(padded_query, query_length)
        l_mask = self._gen_lan_mask(self.max_length,query_length).to(l_embedded.device)

        #embedding normalization
        if self.unit_vector:
            l_embedded = F.normalize(l_embedded, dim=-1) 
            v_embedded_pos = {k:F.normalize(v_embedded_pos[k], dim=-1) for k in self.keys}
            if v_embedded_neg_intra is not None:
                v_embedded_neg_intra = {k:F.normalize(v_embedded_neg_intra[k], dim=-1) for k in self.keys}
            if v_embedded_neg_inter is not None:
                v_embedded_neg_inter = {k:F.normalize(v_embedded_neg_inter[k], dim=-1) for k in self.keys}
            
        # meta-comparison
        c_pos, c_neg_intra, c_neg_inter = self.compare_emdedded_snippets(
            v_embedded_pos, v_embedded_neg_intra, v_embedded_neg_inter,
            l_embedded, l_mask, visual_pos, visual_neg_intra, visual_neg_inter)
        return c_pos, c_neg_intra, c_neg_inter

    def encode_query(self, padded_query, query_length):
        B = len(padded_query)
        packed_query = pack_padded_sequence(padded_query, 
                                            query_length,
                                            batch_first=True)
        packed_output, _ = self.sentence_encoder(packed_query)
        output, _  = pad_packed_sequence(packed_output, batch_first=True,
                                        total_length=self.max_length)

        # Pass hidden states though a shared linear layer.
        embedded_lang = self.state_encoder(output)

        return embedded_lang

    def encode_visual(self, pos, neg_intra, neg_inter):
        pos, neg_intra, neg_inter = self._unpack_visual(pos, neg_intra, neg_inter)
        embedded_neg_intra, embedded_neg_inter = None, None

        embedded_pos = {k:self.fwd_visual_snippets(k,pos[k]) for k in self.keys}

        mask_key = '-'.join(['mask',self.keys[0]])          
        if neg_intra[mask_key] is not None:
            embedded_neg_intra = {k:self.fwd_visual_snippets(k,neg_intra[k]) for k in self.keys}
                                    
        if neg_inter[mask_key] is not None:
            embedded_neg_inter = {k:self.fwd_visual_snippets(k,neg_inter[k]) for k in self.keys}
                                    
        return embedded_pos, embedded_neg_intra, embedded_neg_inter

    def compare_emdedded_snippets(self, embedded_p, embedded_n_intra,
                                  embedded_n_inter, embedded_a, l_mask,
                                  pos, neg_intra, neg_inter):
        pos, neg_intra, neg_inter = self._unpack_visual(pos, neg_intra, neg_inter)
        c_neg_intra, c_neg_inter = None, None
        mask_p, mask_n_intra, mask_n_inter = {}, {}, {}
        
        for k in self.keys:
            mask_key = '-'.join(['mask',k])
            mask_p[k] = pos[mask_key]
            mask_n_intra[k] = neg_intra[mask_key]
            mask_n_inter[k] = neg_inter[mask_key]
        
        # merge two stream based on mask. 
        embedded_p,embedded_n_intra,embedded_n_inter,mask_p,mask_n_intra,mask_n_inter  = \
            self.merge_features(embedded_p,embedded_n_intra,embedded_n_inter,mask_p,mask_n_intra,mask_n_inter)
        
        c_pos = self.compare_emdeddings(embedded_p, embedded_a,mask_p, l_mask) 

        if mask_n_inter is not None:
            c_neg_intra = self.compare_emdeddings(embedded_n_intra, embedded_a,
                                            mask_n_intra, l_mask)
        if mask_n_inter is not None:
            c_neg_inter = self.compare_emdeddings(embedded_n_inter, embedded_a,
                                            mask_n_inter, l_mask)
        c_pos = {'unified_score': c_pos}
        c_neg_intra = {'unified_score': c_neg_intra}
        c_neg_inter = {'unified_score': c_neg_inter}
        return c_pos, c_neg_intra, c_neg_inter

    def compare_emdeddings(self, v, l, mv, ml):
        return self.chamfer_distance(v, l, mv, ml)

    def merge_features(self, embedded_p,embedded_n_intra,embedded_n_inter,
                                        mask_p,mask_n_intra,mask_n_inter):

        intra_flag = mask_n_intra[self.keys[0]] is not None
        inter_flag = mask_n_inter[self.keys[0]] is not None

        B = embedded_p['rgb'].shape[0]
        F = embedded_p['rgb'].shape[2]
        n1 = int(max(mask_p['rgb'].sum(dim=1)+mask_p['obj'].sum(dim=1)))
        n2 = 0
        if intra_flag:
            n2 = int(max(mask_n_intra['rgb'].sum(dim=1)+mask_n_intra['obj'].sum(dim=1)))
        n3 = 0
        if inter_flag:
            n3 = int(max(mask_n_inter['rgb'].sum(dim=1)+mask_n_inter['obj'].sum(dim=1)))
        T = max(n1,n2,n3)  

        device = embedded_p['rgb'].device
        new_embedded_p = torch.zeros(B,T,F).to(device)
        new_mask_p     = torch.zeros(B,T).to(device)
        elem_rgb_p     = mask_p['rgb'].sum(dim=1)
        elem_obj_p     = mask_p['obj'].sum(dim=1)

        if intra_flag:
            new_embedded_n_intra = torch.zeros(B,T,F).to(device)
            new_mask_n_intra     = torch.zeros(B,T).to(device)
            elem_rgb_n_intra     = mask_n_intra['rgb'].sum(dim=1)
            elem_obj_n_intra     = mask_n_intra['obj'].sum(dim=1)

        if inter_flag:
            new_embedded_n_inter = torch.zeros(B,T,F).to(device)
            new_mask_n_inter     = torch.zeros(B,T).to(device)
            elem_rgb_n_inter     = mask_n_inter['rgb'].sum(dim=1)
            elem_obj_n_inter     = mask_n_inter['obj'].sum(dim=1)
            
        for i in range(B):
            idx1,idx2 = int(elem_rgb_p[i]), int(elem_obj_p[i])
            new_embedded_p[i,:idx1]          = embedded_p['rgb'][i,:idx1]
            new_embedded_p[i,idx1:idx1+idx2] = embedded_p['obj'][i,:idx2]
            new_mask_p[i,:idx1+idx2]         = 1

            if intra_flag:
                idx1,idx2 = int(elem_rgb_n_intra[i]), int(elem_obj_n_intra[i])
                new_embedded_n_intra[i,:idx1]          = embedded_n_intra['rgb'][i,:idx1]
                new_embedded_n_intra[i,idx1:idx1+idx2] = embedded_n_intra['obj'][i,:idx2]
                new_mask_n_intra[i,:idx1+idx2]         = 1
            
            if inter_flag:
                idx1,idx2 = int(elem_rgb_n_inter[i]), int(elem_obj_n_inter[i])
                new_embedded_n_inter[i,:idx1]          = embedded_n_inter['rgb'][i,:idx1]
                new_embedded_n_inter[i,idx1:idx1+idx2] = embedded_n_inter['obj'][i,:idx2]
                new_mask_n_inter[i,:idx1+idx2]         = 1

        return new_embedded_p,new_embedded_n_intra,new_embedded_n_inter,new_mask_p,new_mask_n_intra,new_mask_n_inter

    def merge_features_corpus(self, embedded_p, mask_p):
 
        B = embedded_p['rgb'].shape[0]
        F = embedded_p['rgb'].shape[2]
        T = int(max(mask_p['rgb'].sum(dim=1)+mask_p['obj'].sum(dim=1)))
    
        new_embedded_p = torch.zeros(B,T,F)
        new_mask_p     = torch.zeros(B,T)
        elem_rgb_p     = mask_p['rgb'].sum(dim=1)
        elem_obj_p     = mask_p['obj'].sum(dim=1)
            
        for i in range(B):
            idx1,idx2 = int(elem_rgb_p[i]), int(elem_obj_p[i])
            new_embedded_p[i,:idx1]          = embedded_p['rgb'][i,:idx1]
            new_embedded_p[i,idx1:idx1+idx2] = embedded_p['obj'][i,:idx2]
            new_mask_p[i,:idx1+idx2]         = 1
        
        return new_embedded_p, new_mask_p

    def predict(self, *args):
        "Compute distance between visual and sentence"
        d_pos, *_ = self.forward(*args)
        d_pos = torch.stack([v for _,v in d_pos.items()]).sum(dim=0)
        return d_pos, False

    def _gen_lan_mask(self,num_words,query_length):
        #TO DO: move this task to dataset_untrimmed
        B = query_length.size()[0]              # Batch size
        mask = torch.zeros(B,num_words)         # Mask initialization to zero
        # mask fill in with lenght of each single query
        for i in range(B):
            mask[i,0:query_length[i]] = 1
        return mask

    def search(self, query, query_length, moments):
        """Exhaustive search of query in table

        TODO: batch to avoid out of memory?
        """
        B = moments['rgb'].shape[0]
        query  = query.repeat(B,1,1)
        l_mask = self._gen_lan_mask(self.max_length,query_length).repeat(B,1)

        embeddings  = {k:v for  k,v in moments.items() if 'mask' not in k}
        masks       = {k.split('-')[-1]:v for  k,v in moments.items() if 'mask' in k}
        feats,v_mask= self.merge_features_corpus(embeddings, masks)

        chamfer_distance = self.compare_emdeddings(feats, query, v_mask, l_mask)
     
        return chamfer_distance, False

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