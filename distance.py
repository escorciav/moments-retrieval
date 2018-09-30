from torch.nn.functional import cosine_similarity


def cosine(x1, x2, dim=1, eps=1e-8):
    "cosine distance between dim of two tensors"
    return 1 - cosine_similarity(x1, x2, dim, eps)


def squared_l2(x1, x2, dim=1):
    "square of L2 distance between dim of two tensors"
    return (x1 - x2).pow(2).sum(dim=dim)