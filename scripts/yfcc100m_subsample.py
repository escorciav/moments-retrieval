"Unpack and sub-sample of YFCC100M relevant for DiDeMo training"
import os
import random
import tarfile
from pathlib import Path
import pandas as pd
from PIL import Image

FILENAME = Path('../data/interim/yfcc100m/intersect_didemo/under-and-not'
                '-nouns-leq-150_topk-1.csv')
OUTPUT_DIR = Path('/mnt/ssd/tmp/yfcc100m')
ROOT_DIR = Path('/mnt/ilcompf2d1/data/yfcc100m/image')
SAMPLES_PER_TAG = 1000
SEED = 1701
OUTPUT_FILE = Path(f'../data/interim/yfcc100m/intersect_didemo/sample-{SEED}'
                    '_under-and-not-nouns-leq-150_topk-1.csv')


random.seed(SEED)
df = pd.read_csv(FILENAME, index_col=0)
df['adobe_cil_entry'] = df['adobe_cil'].apply(os.path.dirname)

# subsample per NOUNs
df_gbl = df.groupby('topk_tags')
ind = []
for label, df_i in df_gbl:
    ind_i = df_i.index.tolist()
    random.shuffle(ind_i)
    ind.extend(ind_i[:min(len(ind_i), SAMPLES_PER_TAG)])

if not OUTPUT_DIR.exists():
    os.makedirs(OUTPUT_DIR)
df = df.loc[ind, :]
total_images = len(df)
progress = 0
skipped = 0
df_gbep = df.groupby('adobe_cil_entry')
for entry, grouped in df_gbep:
    tar_file = ROOT_DIR / (entry + '.tar')
    reader = tarfile.open(tar_file)
    members = set(reader.getnames())
    for i, row in grouped.iterrows():
        if not row['adobe_cil'] in members:
            progress += 1
            skipped += 1
            continue

        fid = reader.extractfile(row['adobe_cil'])
        assert fid is not None
        img = Image.open(fid).convert('RGB')
        img_file = OUTPUT_DIR / row['adobe_cil'].replace('/', '_')
        img.save(img_file)

        progress += 1
        if progress % (total_images // 10) == 0:
            print(f'[{progress}/{total_images}]')
print('Number of ignored files', skipped)
df.to_csv(OUTPUT_FILE, index=None)