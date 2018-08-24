"""Select a subset of labels from YFCC100M dataset that aligns with DiDeMo

Notes:
    - topk is exposed, but it's partially useless (as we conceived it)
      because the orignal tags from YFCC100M are alphabetized.
"""
import csv
import glob
import json
import sys

import fire
import numpy as np
import pandas as pd
from spacy.lang.en import LEMMA_EXC, LEMMA_INDEX, LEMMA_RULES
from spacy.lemmatizer import Lemmatizer
from tqdm import tqdm

# Those CSV-files are too big and we need this add-on
csv.field_size_limit(sys.maxsize)
# YFCC100M CSV mapping
IMAGE_ID = 0
USER_ID = 1
USER_TAGS = 8
MACHINE_TAGS = 9
IMAGE_URL = 14
PHOTO_OR_VIDEO = 22
# Adobe-CIL YFCC100M data
IMAGES_PER_TAR = 10000
# Magic constants
TRAINABLE_THRESHOLD = 1000
# Data data data ... the dark matter of machine learning
NOUNS_FILE = 'data/interim/didemo/nouns_to_video.json'
VAL_FILE = 'data/raw/val_data.json'
TAG_FREQUENCY_CSV = 'data/interim/yfcc100m/tag_frequency.csv'


def create_vocabulary(underrepresented_threshold=10):
    "Create vocabulary of underrepresented NOUNs to augment"

    # load stats about nouns in DIDEMO
    with open(NOUNS_FILE, 'r') as fid:
        stats = json.load(fid)
        for k, v in stats['nouns_per_subset'].items():
            stats['nouns_per_subset'][k] = set(v)

    # load annotations in val-set
    with open(VAL_FILE, 'r') as fid:
        data = json.load(fid)
    num_val_instances = len(data)
    spanned_annotations = []
    for k, v in stats['annotations_per_subset']['val'].items():
        spanned_annotations.extend(v)
    spanned_annotations = np.unique(spanned_annotations)

    nouns_vocab = set()
    # TODO add val_ids to prioritize verification step
    nouns_and_metadata = {}
    num_nouns_train = len(stats['counts_per_subset']['train'])
    num_nouns_underrepresented = 0
    val_ids_toimpact = []
    val_ids_represented = []
    for k, v in stats['counts_per_subset']['train'].items():
        if v <= underrepresented_threshold:
            num_nouns_underrepresented += 1
            if k in stats['counts_per_subset']['val']:
                val_ids_toimpact.extend(
                    stats['annotations_per_subset']['val'][k])
                nouns_vocab.add(k)
        else:
            if k in stats['counts_per_subset']['val']:
                val_ids_represented.extend(
                    stats['annotations_per_subset']['val'][k])

    num_samples_well_represented = len(np.unique(val_ids_represented))
    nouns_only_val = (stats['nouns_per_subset']['val'] -
                    stats['nouns_per_subset']['train'])
    num_nouns_notrepresented = len(nouns_only_val)
    for k in nouns_only_val:
        val_ids_toimpact.extend(
            stats['annotations_per_subset']['val'][k])
        nouns_vocab.add(k)
    num_samples_to_impact = len(np.unique(val_ids_toimpact))

    print('Num evaluation instances:', num_val_instances)
    print('Spanned instances', len(spanned_annotations))
    print('NOUNs are underrepresented when appear less than',
          underrepresented_threshold + 1)
    print('Total NOUNs in train', num_nouns_train)
    print('NOUNs Under-represented', num_nouns_underrepresented)
    print('NOUNs Unseen during training', num_nouns_notrepresented)
    print('Num descriptions with Under&Unseen NOUNs', num_samples_to_impact)
    print('Num descriptions with Well NOUNs', num_samples_well_represented)
    print('Pctg to impact',
          f'{num_samples_to_impact / num_val_instances:.4f}')
    return nouns_vocab


def main(wildcard='/mnt/ilcompf2d1/data/yfcc100m/yfcc100m_dataset-*',
         underrepresented_threshold=300, topk=1,
         prefix='nouns-under-and-not-leq'):
    print(f'Top-k {topk}')
    print(f'Under-represented threshold {underrepresented_threshold}')

    # Load clean (scrapped) tags
    clean_tags = set()
    with open(TAG_FREQUENCY_CSV, 'r') as fid:
        reader = csv.DictReader(fid, delimiter=',')
        for row in reader:
            clean_tags.add(row['tag'])

    # load a set of interesting NOUNs from DiDeMo
    didemo_nouns = create_vocabulary(underrepresented_threshold)

    # Lemmatizer to deal with plurals
    lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

    # Extract mapping tags 2 image
    tag2image = {}
    images = []
    image_tags_topk_flickr = []
    image_tags_flickr = []
    image_tags_topk = []
    image_tags = []
    image_urls = []
    for filename in tqdm(glob.glob(wildcard)):
        with open(filename, newline='') as csvfile:
            file_index = filename.split('-')[1]
            reader = csv.reader(csvfile, delimiter='\t',
                                quoting=csv.QUOTE_NONE)
            for line_no, row in enumerate(reader):
                if int(row[PHOTO_OR_VIDEO]) != 0 or len(row[USER_TAGS]) == 0:
                    continue

                counter, amen = 0, False
                tags_topk_flickr = []
                tags_topk = []
                for tag in row[USER_TAGS].split(','):
                    lemmatized_tag = lemmatizer(tag, u'NOUN')[0]
                    if lemmatized_tag not in clean_tags:
                        continue
                    if lemmatized_tag not in didemo_nouns:
                        continue
                    # after this point the image will be considered, thus we
                    # can add it to tag2image dict
                    counter += 1
                    tags_topk.append(lemmatized_tag)
                    tags_topk_flickr.append(tag)
                    amen = True
                    if lemmatized_tag in tag2image:
                        tag2image[lemmatized_tag].append(len(images))
                    else:
                        tag2image[lemmatized_tag] = [len(images)]
                    if counter == topk:
                        # according to previous Bryan's project top-5 tags in
                        # YFCC100M have purity >= 60%
                        break
                tags = [i for i in row[USER_TAGS].split(',')
                        if lemmatizer(i, u'NOUN')[0] in clean_tags]

                # I don't need this but just in case
                if not amen:
                    continue
                tar_index = line_no // IMAGES_PER_TAR
                img_index = line_no % IMAGES_PER_TAR
                image_loc = f'{file_index}-{tar_index:03d}/{tar_index:03d}{img_index:04d}.jpg'
                image_url = row[IMAGE_URL]
                images.append(image_loc)
                image_urls.append(image_url)
                image_tags_topk_flickr.append(';'.join(tags_topk_flickr))
                image_tags_flickr.append(row[USER_TAGS].replace(',', ';'))
                image_tags_topk.append(';'.join(tags_topk))
                image_tags.append(';'.join(tags))

    print('Num clean tags:', len(clean_tags))
    with open(NOUNS_FILE, 'r') as fid:
        didemo_stats = json.load(fid)

    trainable_tags = 0
    val_instances = []
    for k, v in tag2image.items():
        if len(v) > TRAINABLE_THRESHOLD:
            trainable_tags += 1
        if k in didemo_stats['annotations_per_subset']['val']:
            val_instances.extend(didemo_stats['annotations_per_subset']['val'][k])
    val_instances = np.unique(val_instances)
    print(f'Top-{topk} gives {len(tag2image)} '
        f'and {trainable_tags} >= {TRAINABLE_THRESHOLD} images')
    print(f'Number of val instances with those NOUNs {len(val_instances)}')

    df = pd.DataFrame([images, image_urls,
                    image_tags_topk, image_tags,
                    image_tags_topk_flickr, image_tags_flickr]).T
    df.columns = ['adobe_cil', 'url', 'topk_tags', 'tags', 'topk_tags_yfcc100m', 'tags_yfcc100m']

    basename = f'{prefix}-{underrepresented_threshold}_topk-{topk}'
    df.to_csv(f'{basename}.csv')
    with open(f'{basename}.json', 'w') as fid:
        json.dump(tag2image, fid)


if __name__ == '__main__':
    fire.Fire()