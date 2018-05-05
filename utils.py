def ship_to(x, device):
    # TODO: clean like default_collate :S
    y = []
    for i in x:
        if i is None:
            y.append(None)
        elif isinstance(i, dict):
            y.append({k: v.to(device) for k, v in i.items()})
        else:
            y.append(i.to(device))
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


if __name__ == '__main__':
    aja = Multimeter(['hi', 'vi', 'tor'])
    aja.update([1, 2, 3])
    aja.update([3, 2, 1])
    print(f'{aja.report()}')