import argparse
from datetime import datetime
import glob

def read_data(dir):
    files = sorted(glob.glob(dir+"/*.log"))
    files = [f for f in files if "corpus" not in f]
    output,empty = [],[]
    for f in files:
        lines = list(open(f, 'r'))
        x = [print_results(l.strip(),f) for l in lines if "INFO:Batch" in l]
        output.append(''.join(x))
        if len(x) == 0:
            title = f.split('/')[-1]
            output.append(f"{title}\n")
            empty.append(title.split('_')[-1].split('.')[0])

    time = datetime.now()
    filename = f'./scripts/data/single_video_{time}.txt'

    with open(filename, 'w') as f:
        for s in output[0::2]:
            f.write(s)
        for s in output[1::2]:
            f.write(s)
        f.write(f'Re-do val: {empty}\n')
   
def print_results(line,filename):
    splits = line[line.rfind("r@1,0.5:"):].split(" ")
    v1=splits[1].split("r")[0]          # r@1,0.5
    v2=splits[2].split("r")[0]          # r@5,0.5
    v3=splits[3].split("r")[0]          # r@1,0.7
    v4=splits[4]                        # r@5,0.7
    title = filename.split('/')[-1]
    string = f"{title}\t{v1}\t{v3}\t{v2}\t{v4}\n"
    return string

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, default='',
                    help='Absolute path to folder from which read results')
    args = parser.parse_args()

    read_data(args.dir)