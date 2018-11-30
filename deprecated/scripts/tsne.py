import ast
import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

FILENAMES = ['data/interim/didemo/resnet152/320x240_max.h5',
             'data/interim/yfcc100m/resnet152/320x240_001.h5']
METADATA = 'data/interim/yfcc100m/intersect_didemo/tsne-metadata_under-and-not-nouns-leq-150_topk-1.tsv'
LOG_DIR = 'scripts/log'
D = 2048

class HDF5sOnlyDatasets():
    def __init__(self, filenames):
        self.files = filenames

    def __getitem__(self, key):
        name, file_ind = key
        with h5py.File(self.files[file_ind], 'r') as fid:
            return fid[name][:]

repo = HDF5sOnlyDatasets(FILENAMES)
df = pd.read_csv(METADATA, sep='\t')
df['time'] = df['time'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else None)
data = np.zeros((len(df), D), dtype=np.float32)
for i, row in df.iterrows():
    src_name = row['key']
    if row['source'] == 'video':
        start, end = row['time']
        feat = repo[src_name, 0]
        data[i, :] = np.mean(feat[start:end + 1, :], axis=0)
    else:
        data[i, :] = repo[src_name, 1][0, :]

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

data = pd.DataFrame(data)
data.to_csv('data_all.tsv', sep='\t', index=None, header=False)

# This work but it runs locally
# embedding_vars = []
# metadata = []
# # TODO: add dimension
# embedding_vars.append(tf.get_variable("all", initializer=data))
# for tag, df_i in df.groupby('label'):
#     ind = df_i.index
#     embedding_vars.append(tf.get_variable(tag, initializer=data[ind, :]))
# embedding_var = embedding_vars[0]

# summary_writer = tf.summary.FileWriter(LOG_DIR)

# # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
# config = projector.ProjectorConfig()
# # You can add multiple embeddings. Here we add only one.
# embedding = config.embeddings.add()
# embedding.tensor_name = embedding_var.name
# # Link this tensor to its metadata file (e.g. labels).
# embedding.metadata_path = os.path.join(os.getcwd(), METADATA)
# # The next line writes a projector_config.pbtxt in the LOG_DIR.
# # TensorBoard will read this file during startup.
# projector.visualize_embeddings(summary_writer, config)

# # Specify where you find the sprite (we will create this later)
# # embedding.sprite.image_path = path_for_mnist_sprites #'mnistdigits.png'
# # embedding.sprite.single_image_dim.extend([28,28])

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
# saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 1)