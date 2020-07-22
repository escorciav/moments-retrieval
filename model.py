import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from chamfer import DoubleMaskedChamferDistance
from random import sample, shuffle
import numpy as np
MOMENT_RETRIEVAL_MODELS = ['MCN', 'SMCN', 'CALChamfer', 'EarlyFusion', 'LateFusion', 'CALChamferV2']

class MCN(nn.Module):
    """MCN model
    TODO:
        try max pooling
        (pr): compare runtime with LSTM cell
    """

    def __init__(self, visual_size={'rgb':4096}, lang_size=300, embedding_size=100,
                 dropout=0.3, max_length=None, visual_hidden=500,
                 lang_hidden=1000, visual_layers=1, lang_layers=2, unit_vector=False,
                 bi_lstm=False, lang_dropout=0.0, alpha=0.0, n_negatives_inter=1,
                 inter_negatives_mode=0):
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
        self.n_negatives_inter = n_negatives_inter
        self.inter_negatives_mode = inter_negatives_mode
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
                visual_neg_intra=None, visual_neg_inter=None, video_indices=None):
        # v_v_embedded_* are tensors of shape [B, D]
        (v_embedded_pos, v_embedded_neg_intra,
         v_embedded_neg_inter) = self.encode_visual(
             visual_pos, visual_neg_intra, visual_neg_inter)

        if not isinstance(padded_query, list):
            padded_query, query_length = [padded_query], [query_length]

        l_embedded = self.encode_query(padded_query, query_length)
        num_lang_pos = len(l_embedded[self.keys[0]])

        c_pos = [{k:self.compare_emdeddings(l_embedded[k][idx], v_embedded_pos[k]) 
                    for k in self.keys} for idx in range(num_lang_pos)]

        c_neg_intra, c_neg_inter = None, None
        
        # condition_neg_intra = v_embedded_neg_intra[self.keys[0]]
        if v_embedded_neg_intra is not None:
            c_neg_intra = [{k:self.compare_emdeddings(l_embedded[k][0], eni[k]) 
                            for k in self.keys} for eni in v_embedded_neg_intra]
            
        # condition_neg_inter = v_embedded_neg_inter[self.keys[0]]
        if v_embedded_neg_inter is not None:
            c_neg_inter = [{k:self.compare_emdeddings(l_embedded[k][0], eni[k]) 
                            for k in self.keys} for eni in v_embedded_neg_inter]
 
        output_dict = {'p'       : c_pos,
                       'n_intra' : c_neg_intra,
                       'n_inter' : c_neg_inter}
        return output_dict

    def encode_visual(self, pos, neg_intra, neg_inter):
        pos, neg_intra, neg_inter = self._unpack_visual(
            pos, neg_intra, neg_inter)
        embedded_neg_intra, embedded_neg_inter = None, None

        embedded_pos = {k:self.visual_encoder[k](v) for k,v in pos.items()}
        if self.unit_vector:
            embedded_pos = {k:F.normalize(embedded_pos[k], dim=-1) for k in self.keys}

        if neg_intra is not None:
            embedded_neg_intra = [{k:self.visual_encoder[k](ni[k]) for k in self.keys}
                                    for ni in neg_intra]
            if self.unit_vector:
                ## Deprecated
                embedded_neg_intra = {k:F.normalize(embedded_neg_intra[k], dim=-1) for k in self.keys}
                                    
        if neg_inter is not None:
            embedded_neg_inter = [{k:self.visual_encoder[k](ni[k]) for k in self.keys}
                                    for ni in neg_inter]                  
            if self.unit_vector:
                ##Deprecated
                embedded_neg_inter = {k:F.normalize(embedded_neg_inter[k], dim=-1) for k in self.keys}  
                                    
        return embedded_pos, embedded_neg_intra, embedded_neg_inter

    def encode_query(self, padded_query, query_length):
        B = len(padded_query[0])
        packed_query = [pack_padded_sequence(p,l,batch_first=True)
                            for p,l in zip(padded_query,query_length)]
        packed_output = {k:[self.sentence_encoder(p) for p in packed_query] for k in self.keys}
        output = {k:[pad_packed_sequence(p[0], batch_first=True,
                        total_length=self.max_length)[0] for p in packed_output[k]] for k in self.keys}
        # TODO: try max-pooling
         # Introduce regularizers on the produced languages
        last_output = {k:[output[k][i][range(B), ql - 1, :] for i,ql in enumerate(query_length)] for k in self.keys}
        embedded_lang = {k:[self.lang_encoder(lo) for lo in last_output[k]] for k in self.keys}
        if self.unit_vector:
            #Deprecated
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
        d_pos = self.forward(*args)['p']
        d_pos = torch.stack([v for _,v in d_pos[0].items()]).sum(dim=0)
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
                visual_neg_intra=None, visual_neg_inter=None, video_indices=None):
        # v_v_embedded_* are tensors of shape [B, N, D]
        (v_embedded_pos, v_embedded_neg_intra,
         v_embedded_neg_inter) = self.encode_visual(
             visual_pos, visual_neg_intra, visual_neg_inter)

        if not isinstance(padded_query, list):
            padded_query, query_length = [padded_query], [query_length]

        l_embedded = self.encode_query(padded_query, query_length)
        # transform l_emdedded into a tensor of shape [B, 1, D]
        l_embedded = {k:[l.unsqueeze(1) for l in l_embedded[k]] for k in self.keys} 

        # meta-comparison
        c_pos, c_neg_intra, c_neg_inter = self.compare_emdedded_snippets(
            v_embedded_pos, v_embedded_neg_intra, v_embedded_neg_inter,
            l_embedded, visual_pos, visual_neg_intra, visual_neg_inter)
        
        output_dict = {'p'               : c_pos,
                       'n_intra'         : c_neg_intra,
                       'n_inter'         : c_neg_inter}

        return output_dict

    def encode_visual(self, pos, neg_intra, neg_inter):
        pos, neg_intra, neg_inter = self._unpack_visual(pos, neg_intra, neg_inter)
        embedded_neg_intra, embedded_neg_inter = None, None

        embedded_pos = {k:self.fwd_visual_snippets(k,pos[k]) for k in self.keys}
                                
        if neg_intra[0]['mask'] is not None:
            embedded_neg_intra = [{k:self.fwd_visual_snippets(k,ni[k]) for k in self.keys}
                                    for ni in neg_intra]
                                    
        if neg_inter[0]['mask'] is not None:
            embedded_neg_inter = [{k:self.fwd_visual_snippets(k,ni[k]) for k in self.keys}
                                    for ni in neg_inter]
                                    
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
                                  embedded_n_inter, l_embedded,
                                  pos, neg_intra, neg_inter):
        pos, neg_intra, neg_inter = self._unpack_visual(pos, neg_intra, neg_inter)
        c_neg_intra, c_neg_inter = None, None

        # Extract masks
        mask_p       = {k: pos['mask'] for k in self.keys}
        mask_n_intra = [{k: ni['mask'] for k in self.keys} for ni in neg_intra]
        mask_n_inter = [{k: ni['mask'] for k in self.keys} for ni in neg_inter]

        #Compute distances
        num_lang_pos = len(l_embedded[self.keys[0]])
        c_pos = [{k:self.pool_compared_snippets(\
                    self.compare_emdeddings(l_embedded[k][idx], embedded_p[k]), mask_p[k]) \
                        for k in self.keys} for idx in range(num_lang_pos)]
    
        if mask_n_intra[0][self.keys[0]] is not None:
            c_neg_intra = [{k:self.pool_compared_snippets(\
                    self.compare_emdeddings(l_embedded[k][0], eni[k]), mni[k]) \
                        for k in self.keys} for eni, mni in zip(embedded_n_intra,mask_n_intra)]
            
        if mask_n_inter[0][self.keys[0]] is not None:
            c_neg_inter = [{k:self.pool_compared_snippets(\
                    self.compare_emdeddings(l_embedded[k][0], eni[k]), mni[k]) \
                        for k in self.keys} for eni, mni in zip(embedded_n_inter,mask_n_inter)]
            # c_neg_inter = {k:self.pool_compared_snippets(
            #     self.compare_emdeddings(l_embedded, embedded_n_inter[k]),mask_n_inter) for k in self.keys}
        return c_pos, c_neg_intra, c_neg_inter

    def search(self, query, table, clips_per_segment, clips_per_segment_list):
        """Exhaustive search of query in table

        TODO: batch to avoid out of memory?
        """
        query_tensor = query[self.keys[0]][0]
        clip_distance = self.compare_emdeddings(
            query_tensor, table).split(clips_per_segment_list)
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
                argout.append([{k:None for k in keys}])
            else:
                argout.append(i)
        return argout


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
                visual_neg_intra=None, visual_neg_inter=None, video_indices=None):
        '''
            emd reffear to embedded.
        '''

        if not isinstance(padded_query, list):
            padded_query, query_length = [padded_query], [query_length]

        # v_v_embedded_* are tensors of shape [B, N, D]
        (v_emb_pos, v_emb_neg_intra,v_emb_neg_inter) = self.encode_visual(
                                    visual_pos, visual_neg_intra, visual_neg_inter)

        l_emb, l_regularizer = self.encode_query(padded_query, query_length)
        l_mask = self._gen_lan_mask(self.max_length,query_length,device=l_emb[self.keys[0]][0].device)

        # meta-comparison
        c_pos, c_neg_intra, c_neg_inter = self.compare_emdedded_snippets(
                                            v_emb_pos, v_emb_neg_intra, v_emb_neg_inter,
                                            l_emb, l_mask, visual_pos, visual_neg_intra, visual_neg_inter)

        output_dict = {'p'               : c_pos,
                       'n_intra'         : c_neg_intra,
                       'n_inter'         : c_neg_inter,
                       'lang_regularizer': l_regularizer}

        return output_dict

    def encode_query(self, padded_query, query_length):
        B = len(padded_query[0])
        packed_query = [pack_padded_sequence(p,l,batch_first=True)
                                    for p,l in zip(padded_query,query_length)]

        packed_output = {k:[self.sentence_encoder[k](p) for p in packed_query] for k in self.keys}
        output = {k:[pad_packed_sequence(p[0], batch_first=True,
                        total_length=self.max_length)[0] for p in packed_output[k]] for k in self.keys}

        # Introduce regularizers on the produced languages
        last_output = {k:[output[k][i][range(B), ql - 1, :] for i,ql in enumerate(query_length)] for k in self.keys}
        embedded_last_layer = {k:[self.state_encoder[k](lo) for lo in last_output[k]] for k in self.keys}
        l_regularizer = {k:self.compare_embedded_lang(embedded_last_layer[k]) for k in self.keys}

        # l_regularizer = {k:[torch.zeros(B) for _ in range(len(output[k]))] for k in self.keys}

        # Pass hidden states though a shared linear layer.
        embedded_lang = {k:[self.state_encoder[k](o) for o in output[k]] for k in self.keys}

        return embedded_lang, l_regularizer

    def encode_visual(self, pos, neg_intra, neg_inter):
        pos, neg_intra, neg_inter = self._unpack_visual(pos, neg_intra, neg_inter)
        embedded_neg_intra, embedded_neg_inter = None, None

        embedded_pos = {k:self.fwd_visual_snippets(k,pos[k]) for k in self.keys}

        mask_key = '-'.join(['mask',self.keys[0]])          
        if neg_intra[0][mask_key] is not None:
            embedded_neg_intra = [{k:self.fwd_visual_snippets(k, ni[k]) for k in self.keys}
                                    for ni in neg_intra]

        if neg_inter[0][mask_key] is not None:
            embedded_neg_inter = [{k: self.fwd_visual_snippets(k, ni[k]) for k in self.keys}
                                    for ni in neg_inter]
                                    
        return embedded_pos, embedded_neg_intra, embedded_neg_inter

    def compare_emdedded_snippets(self, embedded_p, embedded_n_intra,embedded_n_inter,
                                  embedded_a, l_mask, pos, neg_intra, neg_inter):
        pos, neg_intra, neg_inter = self._unpack_visual(pos, neg_intra, neg_inter)
        c_neg_intra, c_neg_inter = None, None

        # Refactor masks
        mask_p       = {k: pos['-'.join(['mask', k])] for k in self.keys}
        mask_n_intra = [{k: ni['-'.join(['mask', k])] for k in self.keys} for ni in neg_intra]
        mask_n_inter = [{k: ni['-'.join(['mask', k])] for k in self.keys} for ni in neg_inter]
        
        c_pos = [{k:self.compare_emdeddings(embedded_p[k], embedded_a[k][idx], mask_p[k], l_mask[idx])
                  for k in self.keys} for idx in range(len(l_mask))]

        # keep only the first tensor from embedded_a[k] and l_mask as it is the one coming from the annotations
        if mask_n_intra[0][self.keys[0]] is not None:
            c_neg_intra = [{k: self.compare_emdeddings(eni[k], embedded_a[k][0], mni[k], l_mask[0]) for k in self.keys}
                           for eni, mni in zip(embedded_n_intra, mask_n_intra)]

        if mask_n_inter[0][self.keys[0]] is not None:
            c_neg_inter = [{k:self.compare_emdeddings(eni[k], embedded_a[k][0], mni[k], l_mask[0]) for k in self.keys}
                            for eni,mni in zip(embedded_n_inter,mask_n_inter)]

        return c_pos, c_neg_intra, c_neg_inter

    def compute_embeddings_norms(self, embedded_p, embedded_n_intra,embedded_n_inter,
                                  embedded_a, l_mask, pos, neg_intra, neg_inter):
        pos, neg_intra, neg_inter = self._unpack_visual(pos, neg_intra, neg_inter)
        norms = {k:[] for k in self.keys}

        # Refactor masks
        mask_p       = {k: pos['-'.join(['mask', k])] for k in self.keys}
        mask_n_intra = [{k: ni['-'.join(['mask', k])] for k in self.keys} for ni in neg_intra]
        mask_n_inter = [{k: ni['-'.join(['mask', k])] for k in self.keys} for ni in neg_inter]
        
        if mask_n_inter[0][self.keys[0]] is not None or mask_n_intra[0][self.keys[0]] is not None:
            for k in self.keys:
                norms[k].append(self.dict_norm(embedded_p[k], mask_p[k]))
                norms[k].append(self.dict_norm(embedded_a[k], l_mask))
                norms[k].append(self.list_norm(embedded_n_intra, mask_n_intra, k))
                norms[k].append(self.list_norm(embedded_n_inter, mask_n_inter, k))
                norms[k] = torch.stack(norms[k]).mean(dim=0)
        return norms

    def dict_norm(self, emb, mask):
        if type(emb) is list:
            norm = torch.norm(torch.stack(emb), p=2, dim=-1)
            mask = [m.sum(dim=-1) for m in mask]
            tmp  = [torch.stack([norm[i,:,:int(idx)].mean() for idx in m]) for i,m in enumerate(mask)] 
            stack_ = torch.stack(tmp)
        else:
            norm = torch.norm(emb, p=2, dim=-1)
            mask = mask.sum(dim=-1)
            stack_ = torch.stack([norm[:,:int(idx)].mean() for idx in mask]) 
        return stack_[~torch.isnan(stack_)].mean(dim=0)

    def list_norm(self, emb, mask, k ):
        norm = torch.norm(torch.stack([e[k] for e in emb]), p=2, dim=-1)
        mask = [m[k].sum(dim=-1) for m in mask]
        tmp  = [torch.stack([norm[i,:,:int(idx)].mean() for idx in m]) for i,m in enumerate(mask)] 
        stack_ = torch.stack(tmp)
        return stack_[~torch.isnan(stack_)].mean(dim=0)
    
    def compare_embedded_lang(self, emb):
        return sum([self.L2Distance(emb[0],e) for e in emb[1:]])

    def compare_emdeddings(self, v, l, mv, ml):
        return self.chamfer_distance(v, l, mv, ml)

    def L2Distance(self, anchor, x, dim=-1):
        y = anchor - x
        return (y * y).sum(dim=dim)

    def _gen_lan_mask(self,num_words,query_length, device):
        #TO DO: move this task to dataset_untrimmed
        B = query_length[0].size()[0]           # Batch size
        mask = []
        for ql in query_length:
            m = torch.zeros(B,num_words,device=device)         # Mask initialization to zero
            # mask fill in with lenght of each single query
            for i in range(B):
                m[i,0:ql[i]] = 1
            mask.append(m)
        return mask

    def _gen_lan_mask_corpus(self, max_num_words, query_length):
        mask = torch.zeros(1, max_num_words)    # Mask initialization to zero
        mask[0, 0:query_length] = 1
        return mask

    def search(self, query, query_length, moments, v_mask, batch_size):
        """Exhaustive search of query in table

        TODO: batch to avoid out of memory?
        """
        if isinstance(query, list):
            query = query[0]
            query_length = query_length[0]

        if batch_size == 0:
            B = moments.shape[0]
            _, d1, d2 = query.size()
            query  = query.expand(B, d1, d2)
            l_mask = self._gen_lan_mask_corpus(self.max_length,query_length)
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
            l_mask = self._gen_lan_mask_corpus(self.max_length,query_length).to('cuda')
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
                argout.append([{k:None for k in keys}])
            else:
                argout.append(i)
        return argout


class CALChamferV2(CALChamfer):
    '''CALChamfer

    Extends CALChamfer by removing inter video data and using intra video negatives as inter video ones within the batch
    '''

    def __init__(self, *args, **kwargs):
        super(CALChamferV2, self).__init__(*args, **kwargs)

    def forward(self, padded_query, query_length, visual_pos,
                visual_neg_intra=None, visual_neg_inter=None, video_indices=None):
        '''
            emd reffear to embedded.
        '''

        if not isinstance(padded_query, list):
            padded_query, query_length = [padded_query], [query_length]

        # v_v_embedded_* are tensors of shape [B, N, D]
        (v_emb_pos, v_emb_neg_intra,v_emb_neg_inter) = self.encode_visual(
                                    visual_pos, visual_neg_intra, visual_neg_inter)

        l_emb, l_regularizer = self.encode_query(padded_query, query_length)
        l_mask = self._gen_lan_mask(self.max_length,query_length,device=l_emb[self.keys[0]][0].device)

        # meta-comparison
        c_pos, c_neg_intra, _ = self.compare_emdedded_snippets(
                    v_emb_pos, v_emb_neg_intra, v_emb_neg_inter,
                    l_emb, l_mask, visual_pos, visual_neg_intra, visual_neg_inter)

        # Compute c_neg_inter using intra negatives within the batch
        c_neg_inter = None
        if v_emb_neg_intra is not None:
            c_neg_inter = self.compare_emdedded_snippets_inter_negatives(
                        v_emb_pos, v_emb_neg_intra,l_emb, l_mask, visual_pos,
                        visual_neg_intra, visual_neg_inter, video_indices)

        output_dict = {'p'       : c_pos,
                       'n_intra' : c_neg_intra,
                       'n_inter' : c_neg_inter}

        return output_dict

    def compare_emdedded_snippets_inter_negatives(self, emb_p, emb_n_intra,
                                  emb_a, l_mask, pos, neg_intra, neg_inter, v_idx):
        pos, neg_intra = self._unpack_visual(pos, neg_intra)
        c_neg_inter =  []

        # preprocess on data to remove useless dimensions. (Only support one intra negative)
        neg_intra = neg_intra[0]
        emb_n_intra = emb_n_intra[0]
        emb_a = {k:v[0] for k,v in emb_a.items()}

        # Refactor masks
        mask_p       = {k: pos['-'.join(['mask', k])] for k in self.keys}
        mask_n_intra = {k: neg_intra['-'.join(['mask', k])] for k in self.keys} 

        # Determine valid inter negatives (must come from a different video)
        valid_idx_per_video = self.filter_invalid_negatives(v_idx)
        devices = {k:emb_n_intra[k].device for k in self.keys}

        # Loop over all examples (batch size - number of samples from the same video)
        for i,v_k in enumerate(v_idx):
            # for each positive in the batch determine the inter samples within the batch and get the valid_idx
            idx = valid_idx_per_video[int(v_k)]
            
            # Select the visual embedding related to those samples
            if self.inter_negatives_mode == 0:
                emb_n  = {k:emb_n_intra[k][idx] for k in self.keys}
                mask_n = {k:mask_n_intra[k][idx] for k in self.keys}

            elif self.inter_negatives_mode == 1:
                emb_n  = {k:emb_p[k][idx] for k in self.keys}
                mask_n = {k:mask_p[k][idx] for k in self.keys}

            elif self.inter_negatives_mode == 2:    
                l = len(idx)//2
                idx = idx[torch.randperm(len(idx))]
                emb_n  = {k:torch.cat((emb_p[k][idx[:l]],
                                       emb_n_intra[k][idx[l:]])) for k in self.keys}
                mask_n = {k:torch.cat((mask_p[k][idx[:l]], 
                                       mask_n_intra[k][idx[l:]])) for k in self.keys}

            elif self.inter_negatives_mode == 3:    
                emb_n  = {k:torch.cat((emb_p[k][idx], 
                                       emb_n_intra[k][idx])) for k in self.keys}
                mask_n = {k:torch.cat((mask_p[k][idx], 
                                       mask_n_intra[k][idx])) for k in self.keys}
            else:
                raise ValueError('Unsuported inter negative mode selection:' + \
                             '0-from intra neg, 1-from pos, 2-from both, 3-concat')
            
            # Expand the positive query to match the size of the inter negatives
            d0,d1,d2 = emb_n['rgb'].shape[0], self.max_length, self.embedding_size
            emb_a_tmp = {k:emb_a[k][i].expand(d0, d1, d2) for k in self.keys}
            # Expand query mask
            mask_a = l_mask[0][i].expand(d0, d1)
            # Compare pos language with negative inter visual embeddings
            c_neg_inter.append({k:self.compare_emdeddings(emb_n[k], emb_a_tmp[k], 
                                        mask_n[k], mask_a) for k in self.keys})

        return self.fix_sizes(c_neg_inter)

    def filter_invalid_negatives(self, v_idx):
        v_idx    = v_idx.tolist()
        list_idx = torch.arange(0, len(v_idx))
        max_neg  = min(self.n_negatives_inter,len(v_idx)-1)
        
        dict_ = {}
        if len(v_idx) == len(set(v_idx)):
            dict_ = {k:list_idx[list_idx!=i] for i,k in enumerate(v_idx)}
        else:
            for i, k in enumerate(v_idx):
                if v_idx.count(k)==1:
                    dict_[k] = list_idx[list_idx!=i]
                else:
                    indices = [ind for ind, x in enumerate(v_idx) if x == k]
                    tmp = list_idx
                    for ind in indices:
                        tmp = tmp[tmp!=ind]
                    tmp = tmp.tolist()
                    tmp.extend(sample(tmp, max(0,max_neg - len(tmp))))
                    dict_[k] = torch.tensor(tmp, dtype=torch.long)
        
        # TODO: Cut down to max number of negatives
        dict_ = {k:v[torch.multinomial(torch.ones(v.shape[0])/v.shape[0], 
                    max_neg, replacement=False).sort().values] for k,v in dict_.items()}

        return dict_

    def fix_sizes(self, scores):
        num_neg  = len(scores[0][self.keys[0]])
        scores_prime = [{k:None for k in self.keys} for _ in range(num_neg)]
        for k in self.keys:
            stack_ = torch.stack([s[k] for s in scores])
            for i in range(stack_.shape[1]):
                scores_prime[i][k] = stack_[:,i] 
        return scores_prime


class EarlyFusion(CALChamfer):
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

        # Language branch
        self.sentence_encoder = nn.ModuleDict({})
        for key in ['shared']:
            self.sentence_encoder[key] = nn.LSTM(
                            self.lang_size, self.lang_hidden, 
                            batch_first=True, bidirectional=self.bi_lstm, 
                            num_layers=1)

        self.state_encoder = nn.ModuleDict({})
        for key in ['shared']:
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

    def forward(self, padded_query, query_length, visual_pos,
                visual_neg_intra=None, visual_neg_inter=None, video_indices=None):
        '''
            emd reffear to embedded.
        '''

        if not isinstance(padded_query, list):
            padded_query, query_length = [padded_query], [query_length]

        # v_v_embedded_* are tensors of shape [B, N, D]
        (v_emb_pos, v_emb_neg_intra,v_emb_neg_inter) = self.encode_visual(
                                    visual_pos, visual_neg_intra, visual_neg_inter)

        l_emb = self.encode_query(padded_query, query_length)
        l_mask = self._gen_lan_mask(self.max_length,query_length,device=l_emb['shared'][0].device)

        # meta-comparison
        c_pos, c_neg_intra, c_neg_inter = self.compare_emdedded_snippets(
                                            v_emb_pos, v_emb_neg_intra, v_emb_neg_inter,
                                            l_emb, l_mask, visual_pos, visual_neg_intra, visual_neg_inter)

        output_dict = {'p'               : c_pos,
                       'n_intra'         : c_neg_intra,
                       'n_inter'         : c_neg_inter}

        return output_dict

    def encode_query(self, padded_query, query_length):
        B = len(padded_query[0])
        packed_query = [pack_padded_sequence(p,l,batch_first=True)
                                    for p,l in zip(padded_query,query_length)]

        packed_output = {k:[self.sentence_encoder[k](p) for p in packed_query] for k in ['shared']}
        output = {k:[pad_packed_sequence(p[0], batch_first=True,
                        total_length=self.max_length)[0] for p in packed_output[k]] for k in ['shared']}

        # Pass hidden states though a shared linear layer.
        embedded_lang = {k:[self.state_encoder[k](o) for o in output[k]] for k in ['shared']}

        return embedded_lang

    def compare_emdedded_snippets(self, embedded_p, embedded_n_intra,
                                  embedded_n_inter, embedded_a, l_mask,
                                  pos, neg_intra, neg_inter):
        pos, neg_intra, neg_inter = self._unpack_visual(pos, neg_intra, neg_inter)
        c_neg_intra, c_neg_inter = None, None
        mask_p, mask_n_intra, mask_n_inter = {}, {}, {}

        assert len(neg_intra)==1, 'Ealy fusion only support 1 intra negative'
        assert len(neg_inter)==1, 'Ealy fusion only support 1 inter negative'
        
        for k in self.keys:
            mask_key = '-'.join(['mask',k])
            mask_p[k] = pos[mask_key]
            mask_n_intra[k] = neg_intra[0][mask_key]
            mask_n_inter[k] = neg_inter[0][mask_key]
        
        # merge two stream based on mask. 
        embedded_p,embedded_n_intra,embedded_n_inter,mask_p,mask_n_intra,mask_n_inter  = \
            self.merge_features(embedded_p,embedded_n_intra,embedded_n_inter,mask_p,mask_n_intra,mask_n_inter)
        
        c_pos = {k:self.compare_emdeddings(embedded_p, embedded_a[k][0], 
                        mask_p, l_mask[0]) for k in ['shared']}

        if mask_n_inter is not None:
            c_neg_intra = {k:self.compare_emdeddings(embedded_n_intra, embedded_a[k][0], 
                                            mask_n_intra, l_mask[0]) for k in ['shared']}
        if mask_n_inter is not None:
            c_neg_inter = {k:self.compare_emdeddings(embedded_n_inter, embedded_a[k][0], 
                                            mask_n_inter, l_mask[0]) for k in ['shared']}


        # Convert scores into list - Depends on the implementation of the loss function. 
        c_pos       = [c_pos]
        c_neg_intra = [c_neg_intra]
        c_neg_inter = [c_neg_inter]

        return c_pos, c_neg_intra, c_neg_inter

    def merge_features(self, embedded_p,embedded_n_intra,embedded_n_inter,
                                        mask_p,mask_n_intra,mask_n_inter):

        new_embedded_p,new_embedded_n_intra,new_embedded_n_inter = None,None,None
        new_mask_p,new_mask_n_intra,new_mask_n_inter = None,None,None

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
                new_embedded_n_intra[i,:idx1]          = embedded_n_intra[0]['rgb'][i,:idx1]
                new_embedded_n_intra[i,idx1:idx1+idx2] = embedded_n_intra[0]['obj'][i,:idx2]
                new_mask_n_intra[i,:idx1+idx2]         = 1
            
            if inter_flag:
                idx1,idx2 = int(elem_rgb_n_inter[i]), int(elem_obj_n_inter[i])
                new_embedded_n_inter[i,:idx1]          = embedded_n_inter[0]['rgb'][i,:idx1]
                new_embedded_n_inter[i,idx1:idx1+idx2] = embedded_n_inter[0]['obj'][i,:idx2]
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

    def search(self, query, query_length, moments):
        """Exhaustive search of query in table
        TODO: batch to avoid out of memory?
        """
        B = moments['rgb'].shape[0]
        query = query['shared'][0]
        _, d1, d2 = query.size()
        query  = query.expand(B, d1, d2)
        l_mask = self._gen_lan_mask_corpus(self.max_length,query_length)
        l_mask = l_mask.expand(B, l_mask.size()[1])

        embeddings  = {k:v for  k,v in moments.items() if 'mask' not in k}
        masks       = {k.split('-')[-1]:v for  k,v in moments.items() if 'mask' in k}
        feats,v_mask= self.merge_features_corpus(embeddings, masks)

        chamfer_distance = self.compare_emdeddings(feats, query, v_mask, l_mask)
     
        return chamfer_distance, False


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