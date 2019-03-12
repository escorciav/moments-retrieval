import glob
import json
import logging
import random
import subprocess
import time

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import default_collate


def collate_data(batch):
    """torchify data during training

    Note: sort to based on query length for using cuDNN-LSTM.
    """
    debug_mode = False
    all_tensors = default_collate(batch)
    if len(all_tensors) == 7:
        debug_mode = True
        idxs, source_ids, *all_tensors = all_tensors
    al_s, idx = all_tensors[1].sort(descending=True)
    a_s = all_tensors[0][idx, ...]
    a_s.requires_grad_()
    dicts_of_tensors = (
        {k: v[idx, ...].requires_grad_() for k, v in i.items()}
        for i in all_tensors[2:])
    argout = (a_s, al_s) + tuple(dicts_of_tensors)
    if debug_mode:
        return (idxs, source_ids) + argout
    return argout


def collate_data_eval(batch):
    """torchify data during eval

    Note: We could do batching but taking care of the length of the sentence
        was a mess, magnified with generic proposal schemes for untrimmed
        videos.
    """
    assert len(batch) == 1
    tensors = []
    for item in batch[0]:
        if isinstance(item, np.ndarray):
            tensors.append(torch.from_numpy(item))
        elif isinstance(item, list):
            tensors.append(torch.tensor(item))
        elif isinstance(item, dict):
            tensors.append({k: torch.from_numpy(t_np)
                            for k, t_np in item.items()})
        else:
            tensors.append(item)
    return tensors


def dict_of_lists(list_of_dicts):
    "Return dict of lists from list of dicts"
    return dict(
        zip(list_of_dicts[0],
            zip(*[d.values() for d in list_of_dicts]))
        )


def dumping_arguments(args, val_performance=None, test_performance=None,
                      perf_per_sample_val=None, perf_per_sample_test=None):
    """Quick-and-dirty way to save args and results

    TODO: put this inside torch.save and khalas!
    """
    if len(args.logfile.name) == 0:
        return
    result_file = args.logfile.with_suffix('.json')
    device = args.device
    topk = args.topk
    # Update dict with performance and remove non-serializable stuff
    if hasattr(args, 'topk_'): delattr(args, 'topk_')
    if hasattr(args, 'writer'): delattr(args, 'writer')
    args.logfile = str(args.logfile)
    args.h5_path = str(args.h5_path) if args.h5_path.exists() else None
    args.train_list = str(args.train_list) if args.train_list.exists() else None
    args.val_list = str(args.val_list) if args.val_list.exists() else None
    args.test_list = str(args.test_list) if args.test_list.exists() else None
    args.snapshot = str(args.snapshot) if args.snapshot.exists() else None
    args.device = None
    args.topk = args.topk.tolist()
    args_dict = vars(args)
    if val_performance is not None:
        args_dict.update({f'val_{k}': v for k, v in val_performance.items()})
    if test_performance is not None:
        args_dict.update({f'test_{k}': v for k, v in test_performance.items()})
    with open(result_file, 'w') as fid:
        json.dump(args_dict, fid, skipkeys=True, indent=1, sort_keys=True)
    if args.dump_results and perf_per_sample_val is not None:
        dump_tensors_as_hdf5(args.logfile + '_instances_rst_val.h5',
                             perf_per_sample_val)
    if args.dump_results and perf_per_sample_test is not None:
        dump_tensors_as_hdf5(args.logfile + '_instances_rst_test.h5',
                             perf_per_sample_test)
    args.device = device
    args.topk = topk


def dump_tensors_as_hdf5(filename, tensors_as_dict_values):
    "Dump dict with torch tensors into a HDF5"
    with h5py.File(filename, 'w') as fid:
        for key, value in tensors_as_dict_values.items():
            fid.create_dataset(name=key, data=value.numpy())


def logfile_from_snapshot(args):
    "Return log-filename for evaluation out of snapshot"
    # remove .pth.tar
    filename = args.snapshot.with_suffix('').with_suffix('')
    # append eval to avoid overwriting
    return filename.with_name(filename.name + '_eval')


def setup_hyperparameters(args):
    "Update Namescope with random hyper-parameters according to a YAML-file"
    if not args.hps:
        return
    filename = args.logfile.parent / 'hps.yml'
    if not filename.exists():
        logging.error(f'Ignoring HPS. Not found {filename}')
        return
    with open(filename, 'r') as fid:
        config = yaml.load(fid)
    logging.info('Proceeding to perform random HPS')
    args_dview = vars(args)

    # Random search over single parameter of tied variables
    slack_tied = {'w_intra': 'w_inter',
                  'c_intra': 'c_inter'}
    for slack, tied in slack_tied.items():
        if tied in config:
            if isinstance(config.get(slack), list):
                logging.warning(f'Ignoring {tied}')
                del config[tied]

    for k, v in config.items():
        if not isinstance(v, list):
            args_dview[k] = v
            continue
        random.shuffle(v)
        args_dview[k] = v[0]
        if k == slack_tied:
            args_dview[slack_tied[k]] = 1 - v[0]

    # Note: only available in YAML
    if args.clip_loss and args.only_clip_loss:
        args_dview['w_intra'] = 0.0
        args_dview['w_inter'] = 0.0


def setup_logging(args):
    "Setup logging to dump progress into file or print it"
    log_prm = dict(format='%(asctime)s:%(levelname)s:%(message)s',
                   level=logging.DEBUG)
    if len(args.logfile.name) >= 1:
        log_prm['filename'] = args.logfile.with_suffix('.log')
        log_prm['filemode'] = 'w'
    logging.basicConfig(**log_prm)
    args.writer = None
    if args.enable_tb:
        # This should be a module variable in case we don't want tensorboard
        args.writer = SummaryWriter(args.logfile.with_suffix(''))


def setup_rng(args):
    "Init random number generators from seed in Namespace"
    if args.seed < 1:
        args.seed = random.randint(0, 2**16)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def save_checkpoint(args, state, record=False):
    "Serialize model into pth"
    if len(args.logfile.name) == 0 or not args.serialize:
        return
    filename = args.logfile.with_suffix('.pth.tar')
    if record:
        epoch = args.epochs
        name = args.logfile.stem
        filename = args.logfile.with_name(f'{name}-{epoch}.pth.tar')
    torch.save(state, filename)


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


def unique2d_perserve_order(x):
    """Return unique along rows in x

    Note: It assumes the same range of numbers for each row in x
    """
    assert x.ndim == 2
    y = []
    for i in range(len(x)):
        _, ind = np.unique(x[i, :], return_index=True)
        y.append(x[i, np.sort(ind)])
    return np.row_stack(y)


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


class Tracker(object):
    "Keep track of torch tensors or numpy scalar things"

    def __init__(self, keys):
        self.data = {i: [] for i in keys}

    def append(self, *args):
        "Add values to track"
        for i, key in enumerate(self.data):
            self.data[key].append(args[i])

    def freeze(self, cpu=True):
        "Make everything a tensor and move to cpu"
        for key, value in self.data.items():
            if not isinstance(value[0], torch.Tensor):
                self.data[key] = torch.tensor(value)
                continue
            elif value[0].dim() > 1 or value[0].shape[0] > 1:
                self.data[key] = torch.stack(value)
            else:
                self.data[key] = torch.cat(value)

            if cpu:
                self.data[key] = self.data[key].to('cpu')
            del (value)


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
