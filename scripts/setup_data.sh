escorcia_folder=/mnt/ilcompf9d1/user/escorcia/moments-retrieval
set -e

# didemo annotations and glove stuff
for i in test_data.json val_data_wwa.json val_data.json test_data_wwa.json train_data_wwa.json train_data.json vocab_glove_complete.txt glove.6B.300d.txt; do
  [ ! -f data/raw/$i ] && ln -s $escorcia_folder/data/raw/$i data/raw/$i
done

# didemo resnet features
didemo_resnet152_dir=data/interim/didemo/resnet152
[ ! -d $didemo_resnet152_dir ] && mkdir $didemo_resnet152_dir
[ ! -f $didemo_resnet152_dir/320x240_max.h5 ] && ln -s $escorcia_folder/$didemo_resnet152_dir/320x240_max.h5 $didemo_resnet152_dir/

# yfcc100m resnet features
yfcc_resnet152_dir=data/interim/yfcc100m/resnet152
[ ! -d $yfcc_resnet152_dir ] && mkdir $yfcc_resnet152_dir
[ ! -f $yfcc_resnet152_dir/320x240_001.h5 ] && ln -s $escorcia_folder/$yfcc_resnet152_dir/320x240_001.h5 $yfcc_resnet152_dir/

# didemo-yfcc100m
filename=data/interim/didemo_yfcc100m/train_data.json
[ ! -d $(dirname $filename) ] && mkdir -p $(dirname $filename)
[ ! -f $filename ] &&  ln -s $escorcia_folder/$i $i

# data for corpus retrieval evaluation
didemo_mcn_dir=data/interim/mcn
[ ! -d $didemo_mcn_dir ] && mkdir $didemo_mcn_dir
for i in corpus_val_rgb.hdf5 queries_val_rgb.hdf5; do
  [ ! -f $didemo_mcn_dir/$i ] && ln -s $escorcia_folder/$didemo_mcn_dir/$i $didemo_mcn_dir/
done

# distance-matrix of smcn model
filename=data/processed/test/smcn_didemo-val_corpus_moment_retrieval.h5
[ ! -f $filename ] && ln -s $escorcia_folder/$filename $filename

# Files for YFCC100M curation
# 1st trag frequency scrapped from previous Bryan project
# 2nd nouns in didemo dataset
for i in data/interim/yfcc100m/tag_frequency.csv data/interim/didemo/nouns_to_video.json; do
  [ ! -f $i ] && ln -s $escorcia_folder/$i $i
done

# pth-model for dumping distance matrix
filename=data/processed/test/hsmcn_10_checkpoint.pth.tar
[ ! -f $filename ] && ln -s $escorcia_folder/$filename $filename

# data for dashboards
gif_dir=dashboards/static/gif
[ ! -d $gif_dir ] && ln -s /mnt/ilcompf9d1/user/escorcia/gif-didemo-val $gif_dir
# moment-retrieval demo
for i in data/interim/mcn/corpus_val_flow.hdf5 data/interim/mcn/rgb-weights.hdf5 data/interim/mcn/flow-weights.hdf5; do
  [ ! -f $i ] && ln -s $escorcia_folder/$i $i
done
# sentence-retrieval
filename=data/processed/test/smcn_12_5_sr_rest.json
[ ! -f $filename ] && ln -s $escorcia_folder/$filename $filename
# moment explorer
filename=data/interim/mcn_retrieval_results/rest_val_intra+inter_rgb+flow.json
[ ! -d $(dirname $filename) ] && mkdir -p $(dirname $filename)
[ ! -f $filename ] && ln -s $escorcia_folder/$filename $filename
