# [:03d].csv

## 001

- comes from intersect_didemo.

- NOUNs under-represented, freq $\leq$ 150, or unseen in the validation set

- sample 1000 random images for each tag

- filter images that were not available

## 002

- similar to 001

- only focus on <=100 nearest neighbors of video descriptors

## 003

### 25-1

- similar to 001

- clean version of 001 via Nearest Neighbors with S2. Check out commit cf5cac36a05cc3037e2c640d51956a6339694615

- parameters
    IMAGES_PER_TAG: 100
    RELAX_FACTOR: 25
    MODE: 1
    MINIMORUM: 75

# train_[:2d]

## 01

- Use images from 001.csv

- Use top-1 tag as description

## 02

- Use images from 001.csv

- Use top-10 tags as description

## 03

- Use images from 002.csv

- Use top-10 tags as description

## 04

- Use images from 003-25-1.csv

- Use top-10 tags as description