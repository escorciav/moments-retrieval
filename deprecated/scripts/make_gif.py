"""Generate GIF for DiDeMo

It takes as input a root folder with the videos of interest.
Each video has a folder with all the sequence of JPG frames, already
pre-processed.
It also need a txt-file, with the videos of interest in the root folder, and
a folder where to place the GIFs.

Notes:
    Dumped all the GIF of the testing set into SSD in 128.9mins with 16 cores.
"""
import datetime
import itertools
import os
from pathlib import Path

import imageio
import numpy as np
from joblib import delayed, Parallel

MIN_MOMENT_DURATION = 5  # duration in seconds
POSSIBLE_SEGMENTS = [(0,0), (1,1), (2,2), (3,3), (4,4), (5,5)]
for i in itertools.combinations(range(6), 2):
    POSSIBLE_SEGMENTS.append(i)


def make_gif(t_start, t_end, video_name, output_dir,
             root=None, fps=5, wh=(320, 240)):
    video_name = Path(video_name).stem
    video_dir = Path(root) / video_name
    output = Path(output_dir) / '{}_{}-{}.gif'.format(video_name,
                                                      t_start, t_end)
    img_files = list(video_dir.glob('*.jpg'))
    img_files.sort()

    f_start = int(t_start * fps)
    f_end = int(t_end * fps)
    if f_start > len(img_files):
        f_start = len(img_files)
    if f_end > len(img_files):
        f_end = len(img_files)
    images_ = [imageio.imread(img_files[i])
               for i in range(f_start, f_end)]

    # padd with zeros if needed
    f_start = int(t_start * fps)
    f_end = int(t_end * fps)
    if len(images_) != (f_end - f_start):
        n_missing = f_end - f_start - len(images_)
        if len(img_files) > 0:
            arr = imageio.imread(img_files[0])
        else:
            arr = np.zeros(wh[::-1] + (3,), dtype=np.uint8)
        images_.extend([np.zeros_like(arr) for i in range(n_missing)])

    # dumping gif
    imageio.mimsave(output, images_)


def dump_all_possible_moments(*args):
    for j, (j_start, j_end) in enumerate(POSSIBLE_SEGMENTS):
        t_start = j_start * MIN_MOMENT_DURATION
        t_end = j_end * MIN_MOMENT_DURATION + MIN_MOMENT_DURATION
        make_gif(t_start, t_end, *args)


def main(filename, output_dir, root, n_jobs=-1, verbose=10):
    print(f'{datetime.datetime.now().isoformat()} Dumping GIFs')
    with open(filename, 'r') as f:
        video_list = [i.strip() for i in f]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    status = Parallel(n_jobs=n_jobs, verbose=verbose, backend='threading')(
        delayed(dump_all_possible_moments)(video_name, output_dir, root)
        for video_name in video_list)

    print(f'{datetime.datetime.now().isoformat()} [Done]')


if __name__ == '__main__':
    # Dump GIFs for a given set
    filename = 'data/raw/videos_test.txt'
    output_dir = '/mnt/ssd/tmp/didemo/gif2'
    root = '/mnt/ssd/tmp/didemo/frames/'
    main(filename, output_dir, root)