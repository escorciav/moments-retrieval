import glob
import json
import random
import time

import pandas as pd
import torch
from torch.utils.data.sampler import Sampler


def jsons_to_dataframe(wilcard):
    "Read multiple json files and stack them into a DataFrame"
    data = []
    for filename in glob.glob(wilcard):
        with open(filename) as f:
            data.append(json.load(f))
    df = pd.DataFrame(data)
    return df


def ship_to(x, device):
    # TODO: clean like default_collate :S
    y = []
    for i in x:
        if isinstance(i, dict):
            y.append({k: v.to(device) for k, v in i.items()})
        elif isinstance(i, torch.Tensor):
            y.append(i.to(device))
        else:
            y.append(i)
    return y


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Multimeter(object):
    "Keep multiple AverageMeter"

    def __init__(self, keys=None):
        self.metrics = keys
        self.meters = [AverageMeter() for i in keys]

    def reset(self):
        for i, _ in enumerate(self.metrics):
            self.meters.reset()

    def update(self, vals, n=1):
        assert len(vals) == len(self.metrics)
        for i, v in enumerate(self.meters):
            v.update(vals[i], n)

    def report(self):
        msg = ''
        for i, v in enumerate(self.metrics):
            msg += f'{v}: {self.meters[i].avg:.4f}\t'
        return msg[:-1]

    def dump(self):
        return {v: self.meters[i].avg for i, v in enumerate(self.metrics)}


class MutableSampler(Sampler):

    def __init__(self, indices=None, num_instances=None):
        assert indices is not None or num_instances is not None
        if num_instances:
            indices = list(range(num_instances))
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        random.shuffle(self.indices)
        return iter(self.indices)

    def set_indices(self, new_indices):
        self.indices = new_indices


def timeit(method):
    """Adapted from Fahim Sakri at Medium
    TODO: remove form final release
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f'{method.__name__}  {te - ts:.2f} s')
        return result
    return timed


if __name__ == '__main__':
    aja = Multimeter(['hi', 'vi', 'tor'])
    aja.update([1, 2, 3])
    aja.update([3, 2, 1])
    print(f'{aja.report()}')