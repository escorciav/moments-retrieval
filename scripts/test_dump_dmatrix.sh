python scripts/moment_retrieval.py \
  --model-pth data/processed/test/hsmcn_10_checkpoint.pth.tar \
  --debug 10
[ -f data/processed/test/hsmcn_10_moment_retrieval.h5 ] && mv data/processed/test/hsmcn_10_moment_retrieval.h5 test_output/data/processed/test/