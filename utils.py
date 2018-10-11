import glob
import json
import logging
import random
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import default_collate


def collate_data(batch):
    """torchify data during training

    Note: sort to based on query length for using cuDNN-LSTM.
    """
    all_tensors = default_collate(batch)
    # for debug (sounds that we should pack it as method in the dataset)
    # idxs, source_ids, *all_tensors = all_tensors
    al_s, idx = all_tensors[1].sort(descending=True)
    a_s = all_tensors[0][idx, ...]
    a_s.requires_grad_()
    dicts_of_tensors = (
        {k: v[idx, ...].requires_grad_() for k, v in i.items()}
        for i in all_tensors[2:])
    argout = (a_s, al_s) + tuple(dicts_of_tensors)
    # for debug
    # return (idxs, source_ids) + argout
    return argout


def collate_test_data(batch):
    """torchify data during eval

    Note: We could do batching but taking care of the length of the sentence
        was a mess, magnified with generic proposal schemes for untrimmed
        videos.
    """
    # for debug (sounds that we should pack it as method in the dataset)
    # ind = 2
    assert len(batch) == 1
    tensors = []
    ind = 0
    for item in batch[0][ind:]:
        if isinstance(item, np.ndarray):
            tensors.append(torch.from_numpy(item))
        elif isinstance(item, list):
            tensors.append(torch.tensor(item))
        elif isinstance(item, dict):
            tensors.append({k: torch.from_numpy(t_np)
                            for k, t_np in item.items()})
        else:
            tensors.append(None)
    return tensors


def dict_of_lists(list_of_dicts):
    "Return dict of lists from list of dicts"
    return dict(
        zip(list_of_dicts[0],
            zip(*[d.values() for d in list_of_dicts]))
        )


def dumping_arguments(args, val_performance, test_performance,
                      performance_per_sample=None, metrics=None):
    "Quick-and-dirty way to save state and results"
    if len(args.logfile) == 0:
        return
    result_file = args.logfile + '.json'
    device = args.device
    # Update dict with performance and remove non-serializable stuff
    args.device = None
    args.rgb_path = str(args.rgb_path)
    args_dict = vars(args)
    args_dict.update({f'val_{k}': v for k, v in val_performance.items()})
    args_dict.update({f'test_{k}': v for k, v in test_performance.items()})
    with open(result_file, 'w') as fid:
        json.dump(args_dict, fid)
    if performance_per_sample is not None:
        with open(args.logfile + '.csv', 'x') as fid:
            fid.write('{},{},{},{}\n'.format('annotation_id', *metrics))
            [fid.write('{},{},{},{}\n'.format(*i))
             for i in performance_per_sample]
    args.device = device


def setup_logging(args):
    "Setup logging to dump progress into file or print it"
    log_prm = dict(format='%(asctime)s:%(levelname)s:%(message)s',
                   level=logging.DEBUG)
    if len(args.logfile) > 1:
        log_prm['filename'] = args.logfile + '.log'
        log_prm['filemode'] = 'w'
    logging.basicConfig(**log_prm)


def setup_rng(args):
    "Init random number generators from seed in Namespace"
    if args.seed < 1:
        args.seed = random.randint(0, 2**16)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def save_checkpoint(args, state):
    "Serialize model into pth"
    if len(args.logfile) == 0 or not args.serialize:
        return
    torch.save(state, args.logfile + '_checkpoint.pth.tar')


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


def get_git_revision_hash():
    "credits: https://stackoverflow.com/a/21901260"
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   universal_newlines=True).strip()


def jsons_to_dataframe(wilcard):
    "Read multiple json files and stack them into a DataFrame"
    data = []
    for filename in glob.glob(wilcard):
        with open(filename) as f:
            data.append(json.load(f))
    df = pd.DataFrame(data)
    return df


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
