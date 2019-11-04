import argparse
import glob

def read_data(dir):
    files = sorted(glob.glob(dir+"/*.log"))
    for f in files:
        lines = list(open(f, 'r'))
        output = [print_results(l.strip(),f) for l in lines if "INFO:Batch" in l]
        if len(output) == 0:
            title = f.split('/')[-1]
            print(f"{title}")

def print_results(line,filename):
    splits = line[line.rfind("r@1,0.5:"):].split(" ")
    v1=splits[1].split("r")[0]          # r@1,0.5
    v2=splits[2].split("r")[0]          # r@5,0.5
    v3=splits[3].split("r")[0]          # r@1,0.7
    v4=splits[4]                        # r@5,0.7
    title = filename.split('/')[-1]
    print(f"{title}\t {v1} {v3} {v2} {v4}")
    return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, default='',
                    help='Absolute path to folder from which read results')
    args = parser.parse_args()

    read_data(args.dir)