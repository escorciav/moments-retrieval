'''
Author: Mattia Soldan
e-mail: mattia.soldan@kaust.edu.sa
'''

from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
import time
import collections


class ChamferDistance(nn.Module):
    "Chamfer distance"

    def __init__(self):
        super(ChamferDistance, self).__init__()   

    def forward(self,video_feat,lang_feat):
        #pairwise distances matrix, shape = [B,Nv,Nl]
        pairwise_dist = self.batch_pairwise_dist(video_feat,lang_feat)
    
        # Normalization values
        Nv = pairwise_dist.size()[1]
        Nl = pairwise_dist.size()[2]

        # language_to_clip
        mins, _ = pairwise_dist.min(dim=2)
        language_to_clip_dist = mins.sum(dim=1)/Nv

        # clip_to_language
        mins, _ = pairwise_dist.min(dim=1)
        clip_to_language_dist = mins.sum(dim=1)/Nl

        return language_to_clip_dist + clip_to_language_dist

    def batch_pairwise_dist(self,x,y):
        # implement the formula 
        # Dij = ||x-y||**2 
        # Expanding it as: Dij = ||x||**2 + ||y||**2 - 2<x,y>

        a = x.pow(2).sum(dim=2,keepdim=True)
        b = y.pow(2).sum(dim=2,keepdim=True)
        ab = torch.bmm(x,y.transpose(2,1))
        pairwise_dist = a - 2*ab + b.transpose(2,1)
        return pairwise_dist


class MaskedChamferDistance(nn.Module):
    "Chamfer distance"

    def __init__(self):
        super(MaskedChamferDistance, self).__init__()   

    def forward(self,video_feat,lang_feat,mask):
        
        mask.detach()   #control gradient flow

        #pairwise distances matrix, shape = [B,Nv,Nl]
        pairwise_dist = self.batch_pairwise_dist(video_feat,lang_feat)
        masked_pairwise_dist = self.masking(pairwise_dist.clone(),mask)

        # Normalization values
        Nv = torch.sum(mask,dim=1)
        Nl = pairwise_dist.size()[2]

        # clip_to_language
        mins, _ = masked_pairwise_dist.min(dim=2)
        masked_min = mins * mask
        clip_to_language_dist = masked_min.sum(dim=1)/Nv

        # language_to_clip
        mins, _ = masked_pairwise_dist.min(dim=1)
        language_to_clip_dist = mins.sum(dim=1)/Nl

        return language_to_clip_dist + clip_to_language_dist

    def batch_pairwise_dist(self,x,y):
        # implement the formula 
        # Dij = ||x-y||**2 
        # Expanding it as: Dij = ||x||**2 + ||y||**2 - 2<x,y>
        a = x.pow(2).sum(dim=2,keepdim=True)
        b = y.pow(2).sum(dim=2,keepdim=True)
        ab = torch.bmm(x,y.transpose(2,1))
        pairwise_dist = a - 2*ab + b.transpose(2,1)
        return pairwise_dist  

    def masking(self,p,m):
        #make negative mask of the same shape of pairwise matrix
        neg_m = m.clone().unsqueeze(-1).expand_as(p) 

        # Negate mask 
        neg_m[neg_m == 0] = 2
        neg_m = neg_m - 1

        #Add maximum value of pairwise matrix to all
        # elements in position given by negate mask 
        max_ = p.max().detach()
        p = p + neg_m * max_
        
        return p
    

class DoubleMaskedChamferDistance(MaskedChamferDistance):
    "Chamfer distance"

    def __init__(self):
        super(DoubleMaskedChamferDistance, self).__init__()   

    def forward(self,video_feat,lang_feat,mask_v,mask_l):
        
        mask_v=mask_v.detach()#.to(minsv.device)  #control gradient flow
        mask_l=mask_l.detach()#.to(minsv.device)

        #pairwise distances matrix, shape = [B,Nv,Nl]
        pairwise_dist = self.batch_pairwise_dist(video_feat,lang_feat)
        masked_pairwise_dist = self.masking(pairwise_dist.clone(),mask_v,mask_l)

        # Normalization values
        Nv = torch.sum(mask_v,dim=1)
        Nl = torch.sum(mask_l,dim=1) 

        # clip_to_language
        minsv, _ = masked_pairwise_dist.min(dim=2)
        masked_minv = minsv * mask_v
        clip_to_language_dist = masked_minv.sum(dim=1)/Nv

        # language_to_clip
        minsl, _ = masked_pairwise_dist.min(dim=1)
        masked_minl = minsl * mask_l
        language_to_clip_dist = masked_minl.sum(dim=1)/Nl

        return (language_to_clip_dist + clip_to_language_dist).cpu()

    def masking(self,p,mv,ml):
        #make negative mask of the same shape of pairwise matrix
        import time 
        t = time.time()
        neg_mv = (mv.clone().unsqueeze(-1).expand_as(p)).cpu().numpy() 
        neg_ml = (ml.clone().unsqueeze(1).expand_as(p)).cpu().numpy()
        # sum and negate masks
        neg = np.logical_not(np.logical_and(neg_ml,neg_mv)).astype(int)
    
        #Add maximum value of pairwise matrix to all
        # elements in position given by negate mask 
        max_ = p.max().detach()
        p = p + (neg * max_).type_as(p)
        
        return p
    
# Deprecated since merge with corpus, I needed to remove the delta variable
class MaskedChamferDistanceVisual(MaskedChamferDistance):
    "Chamfer distance"

    def __init__(self, writer=None):
        super(MaskedChamferDistanceVisual, self).__init__()   
        self.writer = writer
        self.counter = 0

    def forward(self,video_feat,lang_feat,mask,delta):
        
        mask.detach()   #control gradient flow

        #pairwise distances matrix, shape = [B,Nv,Nl]
        pairwise_dist = self.batch_pairwise_dist(video_feat,lang_feat)
        masked_pairwise_dist = self.masking(pairwise_dist.clone(),mask)

        # Normalization values
        Nv = torch.sum(mask,dim=1)
        Nl = pairwise_dist.size()[2]

        # clip_to_language
        mins, idx1 = masked_pairwise_dist.min(dim=2)
        masked_min = mins * mask
        clip_to_language_dist = masked_min.sum(dim=1)/Nv

        # language_to_clip
        mins, idx2 = masked_pairwise_dist.min(dim=1)
        language_to_clip_dist = mins.sum(dim=1)/Nl

        if self.writer is not None:
            self.log_histogram(idx1,idx2,delta)

        return language_to_clip_dist + clip_to_language_dist
    
    def masking(self,p,m):
        #make negative mask of the same shape of pairwise matrix
        neg_m = m.clone().unsqueeze(-1).expand_as(p) 

        # Negate mask 
        neg_m[neg_m == 0] = 2
        neg_m = neg_m - 1

        #Add maximum value of pairwise matrix to all
        # elements in position given by negate mask 
        max_ = p.max().detach()
        p = p + neg_m * max_
        
        return p

    def log_histogram(self, idx1, idx2, delta):

        # v_counter = collections.Counter(list(idx1.cpu().numpy().reshape(-1)))
        # l_counter = collections.Counter(list(idx2.cpu().numpy().reshape(-1)))

        idx2 = (idx2 + delta.unsqueeze(1)) % idx1.size()[1]

        self.writer.add_histogram("chamfer/video_min", idx1.view(-1),self.counter)
        self.writer.add_histogram("chamfer/language_min", idx2.view(-1),self.counter)
        self.counter += 1

# Deprecated since merge with corpus, I needed to remove the delta variable
class DoubleMaskedChamferDistanceVisual(MaskedChamferDistance):
    "Chamfer distance"

    def __init__(self, writer=None):
        super(DoubleMaskedChamferDistanceVisual, self).__init__()   
        self.writer = writer
        self.counter=0

    def forward(self,video_feat,lang_feat,mask_v,mask_l,delta, name, training):
        
        mask_v=mask_v.detach()   #control gradient flow
        mask_l=mask_l.detach()

        #pairwise distances matrix, shape = [B,Nv,Nl]
        pairwise_dist = self.batch_pairwise_dist(video_feat,lang_feat)
        masked_pairwise_dist = self.masking(pairwise_dist.clone(),mask_v,mask_l)

        # Normalization values
        Nv = torch.sum(mask_v,dim=1)
        Nl = torch.sum(mask_l,dim=1) 

        # clip_to_language
        minsv, idxv = masked_pairwise_dist.min(dim=2)
        masked_minv = minsv * mask_v
        clip_to_language_dist = masked_minv.sum(dim=1)/Nv

        # language_to_clip
        minsl, idxl = masked_pairwise_dist.min(dim=1)
        masked_minl = minsl * mask_l
        language_to_clip_dist = masked_minl.sum(dim=1)/Nl

        if self.writer is not None and self.training:             
            v = language_to_clip_dist + clip_to_language_dist     
            self.log_histogram(idxv,idxl,mask_v,mask_l,delta, v, name)

        return language_to_clip_dist + clip_to_language_dist

    def masking(self,p,mv,ml):
        #make negative mask of the same shape of pairwise matrix
        import time 
        t = time.time()
        neg_mv = (mv.clone().unsqueeze(-1).expand_as(p)).cpu().numpy() 
        neg_ml = (ml.clone().unsqueeze(1).expand_as(p)).cpu().numpy()
        # sum and negate masks
        neg = np.logical_not(np.logical_and(neg_ml,neg_mv)).astype(int)
    
        #Add maximum value of pairwise matrix to all
        # elements in position given by negate mask 
        max_ = p.max().detach()
        p = p + (neg * max_).type_as(p)
        
        return p.cuda()
    
    def log_histogram(self, idxv, idxl, maskv, maskl, delta, v, name):
        B = len(delta)
        Nv = idxv.size()[-1]
        Nl = idxl.size()[-1]
        maskv= maskv.sum(dim=1)
        maskl= maskl.sum(dim=1)

        from_vid_to_lang, from_lang_to_vid = [], []
        for i in range(B):
            from_vid_to_lang.append(idxv[i][:int(maskv[i])])
            val = (idxl[i][0:int(maskl[i])] + delta[i]) % Nv
            from_lang_to_vid.append(val)

        from_vid_to_lang= torch.cat(from_vid_to_lang, dim=0)
        from_lang_to_vid= torch.cat(from_lang_to_vid, dim=0)

        self.writer.add_histogram("chamfer/video_to_lang/{}".format(name), from_vid_to_lang.view(-1),self.counter)
        self.writer.add_histogram("chamfer/lang_to_video/{}".format(name),  from_lang_to_vid.view(-1),self.counter)
        self.writer.add_histogram("chamfer_distance/{}".format(name),  v.view(-1),self.counter)
        self.writer.add_scalar("chamfer_distance_mean/{}".format(name),  v.mean(),self.counter)
        self.counter += 1


def _ChamferDistance_v1(vf, lf):

    l1 = 0
    for l in lf:
        d = []
        for v in vf:
            d.append(((l-v)**2).sum(-1))
        l1+= min(d)

    l2 = 0
    for v in vf:
        d = []
        for l in lf:
            d.append(((l-v)**2).sum(-1))
        l2+= min(d)

    return l1/len(lf) + l2/len(vf)


def _ChamferDistance_v2(vf, lf):

    l1 = sum([min([((l-v)**2).sum(-1) for v in vf]) for l in lf])
    l2 = sum([min([((l-v)**2).sum(-1) for l in lf]) for v in vf])

    return l1/len(lf) + l2/len(vf)


def _ChamferDistance_v3(x, y):

    A, _ = x.shape
    B, _ = y.shape

    a = (x**2).sum(axis=1,keepdims=True)
    b = (y**2).sum(axis=1,keepdims=True)
    ab = np.matmul(x,y.T)

    out = a -2*ab +b.T
    l = out.min(axis=1).sum()/A + out.min(axis=0).sum()/B

    return l


def _ChamferDistance_v4(x, y):
    Nl, Nv = x.shape[1],y.shape[1]       #BATCH SIZE, Nv, D
     
    a = (x**2).sum(axis=2,keepdims=True)
    b = (y**2).sum(axis=2,keepdims=True)
    ab = np.matmul(x,y.transpose((0,2,1)))
    out = a -2*ab +b.transpose((0,2,1))

    l = out.min(axis=2).sum(axis=1)/Nl + out.min(axis=1).sum(axis=1)/Nv

    return l


def _comparison_test(video_feat,lang_feat,mask):
    lang_feat2 = lang_feat.detach().cpu().numpy()
    video_feat2 = video_feat.detach().cpu().numpy()
    print("Shapes (D,Nx,D): video={}, lang={}\n".format( video_feat.shape,lang_feat.shape))

    ############################################

    t = time.time()
    l = [_ChamferDistance_v1(video_feat2[i], lang_feat2[i]) for i in range(video_feat.shape[0])]
    print("Avg Distance v1: \t\t\t{:.5f} - {:.5f} sec.\n".format(np.mean(l),time.time()-t))

    ############################################

    t = time.time()
    l = [_ChamferDistance_v2(video_feat2[i],lang_feat2[i]) for i in range(video_feat.shape[0])]
    print("Avg Distance v2: \t\t\t{:.5f} - {:.5f} sec.\n".format(np.mean(l),time.time()-t))

    ############################################

    t = time.time()
    l = [_ChamferDistance_v3(video_feat2[i],lang_feat2[i]) for i in range(video_feat.shape[0])]
    print("Avg Distance v3: \t\t\t{:.5f} - {:.5f} sec.\n".format(np.mean(l),time.time()-t))

    ############################################

    t = time.time()
    l = _ChamferDistance_v4(video_feat2,lang_feat2)
    print("Avg Distance v4: \t\t\t{:.5f} - {:.5f} sec.\n".format(np.mean(l),time.time()-t))

    ############################################

    d = ChamferDistance()
    t = time.time()
    l = d(video_feat, lang_feat)
    print("Avg Distance Chamfer class: \t\t{:.5f} - {:.5f} sec.\n".format(torch.mean(l),time.time()-t))

    ############################################

    d = MaskedChamferDistance()
    t = time.time()
    l = d(video_feat, lang_feat,mask)
    print("Avg Distance Masked Chamfer class: \t{:.5f} - {:.5f} sec.\n".format(torch.mean(l),time.time()-t))

    ############################################

    import silvios_chamfer
    d = silvios_chamfer.ChamferLoss()
    t = time.time()
    l = d(video_feat.transpose(2,1), lang_feat.transpose(2,1))
    print("Avg Distance Silvio's Chamfer: \t\t{:.5f} - {:.5f} sec.\n".format(torch.mean(l),time.time()-t))
        
    ############################################

    #atlas implementation
    # Distance = dist_chamfer.chamferDist()
    # t = time.time()

    # l = Distance(video_feat, lang_feat)
    # print("Distance v5: {:.5f} - {:.5f} sec.\n".format(float(l),time.time()-t))


def _grad_test(video_feat,lang_feat):
    ## Chamfer Distance
    d = ChamferDistance()
    l = d(video_feat, lang_feat)
    
    # check if it possible to use backpropagation
    torch.mean(l).backward()
    if video_feat.grad.size() and lang_feat.grad.size():
        print("Gradiet backprop possible for Chamfer Distance.\n")
    else:
        print("Cannot comput gradients for Chamfer Distance.\n")

    ## Masked Chamfer Distance
    d = MaskedChamferDistance()
    l = d(video_feat, lang_feat, mask)

    # check if it possible to use backpropagation
    torch.mean(l).backward()
    if video_feat.grad.size() and lang_feat.grad.size():
        print("Gradiet backprop possible for Masked Chamfer Distance.\n")
    else:
        print("Cannot comput gradients for Masked Chamfer Distance.\n")


def _gen_mask(B, Nv, all_ones):
    if all_ones:
        return torch.ones((B,Nv)).cuda()
    else:
        mask = torch.randint(0,2,(B,Nv)).cuda()

        for i in range(mask.size()[0]):
            if torch.sum(mask[i])==0:
                mask[i][0]=1
        return mask
  

if __name__ == '__main__':
    ##parameters
    B, D = 128, 100     # batch size, feat size
    Nv, Nl = 5, 3       # num video feat, num lang feat

    ##random vectors
    lang_feat = torch.rand((B,Nl,D)).cuda()
    video_feat = torch.rand((B,Nv,D)).cuda()
    mask = _gen_mask(B, Nv, all_ones=True)

    ##test unit
    lang_feat  = Variable(lang_feat, requires_grad=True)
    video_feat = Variable(video_feat,requires_grad=True)

    _comparison_test(video_feat,lang_feat,mask)
    _grad_test(video_feat, lang_feat)
    