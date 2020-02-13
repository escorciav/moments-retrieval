import argparse
import glob
from datetime import datetime

def read_data(dir):
    files = sorted(glob.glob(dir+"/*.log"))
    files = sorted([f for f in files if 'corpus' in f])
    val = []
    test = []
    empty_val, empty_test = [], []
    for f in files:
        lines = list(open(f, 'r'))
        info = {}
        info['MedRank@0.5']     = ''.join([l.split(': ')[-1] for l in lines if 'MedRank@0.5' in l]).strip()
        info['Recall@1,0.5']    = ''.join([l.split(': ')[-1] for l in lines if 'Recall@1,0.5' in l]).strip()
        info['Recall@10,0.5']   = ''.join([l.split(': ')[-1] for l in lines if 'Recall@10,0.5' in l]).strip()
        info['Recall@100,0.5']  = ''.join([l.split(': ')[-1] for l in lines if 'Recall@100,0.5' in l]).strip()
        info['Recall@1000,0.5'] = ''.join([l.split(': ')[-1] for l in lines if 'Recall@1000,0.5' in l]).strip()
        info['Recall@10000,0.5']= ''.join([l.split(': ')[-1] for l in lines if 'Recall@10000,0.5' in l]).strip()

        info['MedRank@0.7']     = ''.join([l.split(': ')[-1] for l in lines if 'MedRank@0.7' in l]).strip()
        info['Recall@1,0.7']    = ''.join([l.split(': ')[-1] for l in lines if 'Recall@1,0.7' in l]).strip()
        info['Recall@10,0.7']   = ''.join([l.split(': ')[-1] for l in lines if 'Recall@10,0.7' in l]).strip()
        info['Recall@100,0.7']  = ''.join([l.split(': ')[-1] for l in lines if 'Recall@100,0.7' in l]).strip()
        info['Recall@1000,0.7'] = ''.join([l.split(': ')[-1] for l in lines if 'Recall@1000,0.7' in l]).strip()
        info['Recall@10000,0.7']= ''.join([l.split(': ')[-1] for l in lines if 'Recall@10000,0.7' in l]).strip()
        
        info['FileName']        = f.split('/')[-1]    

        string = []
        string.append(f"{info['FileName']}\t")
        string.append(f"{info['Recall@1,0.5']}\t")
        string.append(f"{info['Recall@10,0.5']}\t")
        string.append(f"{info['Recall@100,0.5']}\t")
        string.append(f"{info['Recall@1000,0.5']}\t")
        string.append(f"{info['Recall@10000,0.5']}\t")
        string.append(f"{info['MedRank@0.5']}\t")
        
        string.append(f"{info['Recall@1,0.7']}\t")
        string.append(f"{info['Recall@10,0.7']}\t")
        string.append(f"{info['Recall@100,0.7']}\t")
        string.append(f"{info['Recall@1000,0.7']}\t")
        string.append(f"{info['Recall@10000,0.7']}\t")
        string.append(f"{info['MedRank@0.7']}\t")

        string = (''.join(string)).strip('\n')
        if 'val' in info['FileName']:
            val.append(string+'\n')
            if info['Recall@1,0.5'] == '':
                empty_val.append(info['FileName'].split('_')[-3])
        else:
            test.append(string+'\n')
            if info['Recall@1,0.5'] == '':
                empty_test.append(info['FileName'].split('_')[-3])
        
    time = datetime.now()
    filename = f'./scripts/data/corpus_{time}.txt'
    with open(filename, 'w') as f:
        for s in val:
            f.write(s)
        for s in test:
            f.write(s)
        f.write(f'Re-do val: {empty_val}\n')
        f.write(f'Re-do test: {empty_test}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, default='',
                    help='Absolute path to folder from which read results')
    args = parser.parse_args()

    read_data(args.dir)

