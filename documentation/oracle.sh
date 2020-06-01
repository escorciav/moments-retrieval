
oracle
dataset=data/processed/didemo
interim=data/interim/didemo/best_models/stal_01/a/
concepts=concepts_map_all_words.json

parameters="
--test-list  $dataset/test-01.json 
--h5-path    $dataset/resnet152_rgb_max_cl-2.5.h5 $dataset/obj_predictions_perc_50_glove_bb_spatial.h5
--oracle-map $dataset/concepts/$concepts
--snapshot   $interim/1.json
--logfile    $interim/test_oracle.json
--tags          rgb obj
--snapshot-tags rgb obj
--concepts-oracle 0
--dump --n-display 0.2 "

python corpus_retrieval_eval.py $parameters



