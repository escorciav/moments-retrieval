import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from didemo import Didemo
from model import MCN
from loss import IntraInterRanking
from evaluation import video_evaluation
from utils import Multimeter

RAW_PATH = Path('data/raw')
MODALITY = ['rgb', 'flow']
EVAL_BATCH_SIZE = 1

parser = argparse.ArgumentParser(description='DiDeMo')

parser.add_argument('--lr', type=float, default=0.05,
                            help='initial learning rate')
parser.add_argument('--epochs', type=int, default=50,
                            help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128,
                            help='batch size')
parser.add_argument('--margin', type=float, default=0.1,
                            help='MaxMargin margin value')
parser.add_argument('--lr_decay', type=float, default=0.95,
                            help='Learning rate exp epoch decay')
parser.add_argument('--n_display', type=int, default=60,
                            help='Information display frequence')

parser.add_argument('--gpu-id', type=int, default=-1,
                    help='Use of GPU')
parser.add_argument('--n_cpu', type=int, default=1,
                    help='Number of CPU')

parser.add_argument('--feat', default='rgb', choices=MODALITY,
                    help='kind of modality')
# TODO: enable
# parser.add_argument('--text_cluster_size', type=int, default=32,
#                             help='Text cluster size')
parser.add_argument('--model_name', type=str, default='test',
                            help='Model name')
parser.add_argument('--seed', type=int, default=1,
                            help='Initial Random Seed')

parser.add_argument('--optimizer', type=str, default='adam',
                            help='optimizer')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Nesterov Momentum for SGD')

args = parser.parse_args()
args.device = 'cpu'
if args.gpu_id >= 0 and torch.cuda.is_available():
    args.device = torch.device(f'cuda:{args.gpu_id}')
print(args)

rgb_feat_path = RAW_PATH / 'average_fc7.h5'
flow_feat_path = RAW_PATH / 'average_global_flow.h5'
train_list_path = RAW_PATH / 'train_data.json'
val_list_path = RAW_PATH / 'val_data.json'

if args.feat == 'rgb':
    flow_feat_path = None
else:
    rgb_feat_path = None

# Predefining random initial seeds
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# random.seed(args.seed)

print('Pre-loading features... This may take a couple of minutes.')
train_dataset = Didemo(train_list_path, rgb_file=rgb_feat_path,
                       flow_file=flow_feat_path)
                       # loc=True, max_words=50)
val_dataset = Didemo(val_list_path, rgb_file=rgb_feat_path,
                     flow_file=flow_feat_path, test=True)

# Setup data loaders
print('Setting-up loaders')
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.n_cpu,
                              collate_fn=train_dataset.collate_data)
val_dataloader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE,
                            shuffle=False, num_workers=args.n_cpu,
                            collate_fn=val_dataset.collate_test_data)

# Model
print('Setting-up model')
feat_0 = train_dataset[0]
text_dim = feat_0[0].shape[1]
video_modality_dim = feat_0[2].shape[0]
max_length = feat_0[0].shape[0]
net = MCN(visual_size=video_modality_dim, lang_size=text_dim,
          max_length=max_length)
net.train()
if args.gpu_id >= 0:
    print('Transferring model to GPU')
    net.to(args.device)

# Criterion
print('Setting-up criterion')
ranking_loss = IntraInterRanking(margin=args.margin)
if args.gpu_id >= 0:
    print('Transferring criterion to GPU')
    ranking_loss.to(args.device)

# Optimizer
print('Setting-up optimizer')
if args.optimizer == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.optimizer == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum)

print('Starting optimization...')
for epoch in range(args.epochs):
    running_loss = 0.0
    print(f'Epoch: {epoch + 1}')

    # TODO: wrap it into a function
    for it, minibatch in enumerate(train_dataloader):
        if args.gpu_id >= 0:
            minibatch_ = minibatch
            minibatch = [i.to(args.device) for i in minibatch]

        embeddings = net(*minibatch)
        loss, _, _ = ranking_loss(*embeddings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (it + 1) % args.n_display == 0:
            print(f'Epoch: {epoch + 1}, '
                  f'status: {it / len(train_dataloader):.2f}, '
                  f'loss: {running_loss / args.n_display:.4f}')
            running_loss = 0.0

    # TODO: wrap it into a function
    print(f'* Evaluation')
    meters = Multimeter(keys=['iou', 'r@1', 'r@5'])
    net.eval()
    for it, minibatch in enumerate(val_dataloader):
        if args.gpu_id >= 0:
            minibatch_ = minibatch
            minibatch = [i.to(args.device) if i is not None else None
                         for i in minibatch]

        l_embedding, v_embedding, *_ = net(*minibatch)
        distance = (l_embedding - v_embedding).pow(2).sum(dim=1)
        # TODO: port evaluation to GPU
        _, idx = distance.sort(descending=False)
        idx_h = idx.to('cpu')
        predictions = [val_dataset.segments[i] for i in idx_h]
        gt = val_dataset.metadata[it]['times']
        performance = video_evaluation(gt, predictions)
        meters.update(performance)
    print(f'{meters.report()}')
    net.train()

    # TODO: wrap it into a function
    for param_group in optimizer.param_groups:
        param_group['lr'] *= args.lr_decay