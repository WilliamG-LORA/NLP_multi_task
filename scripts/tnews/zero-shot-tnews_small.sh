#!/bin/bash

# bash scripts/zero-shot-tnews_small.sh


CHECKPOINT_PATH="../CPM-distill"
EVALDATA_PATH="../nlpdata/tnews_public"

MPSIZE=2
NLAYERS=12
NHIDDEN=768
NATT=12
MAXSEQLEN=1024

CMD="python -m torch.distributed.launch --nproc_per_node $MPSIZE zero-shot_tnews.py \
       --model-parallel-size $MPSIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --load $CHECKPOINT_PATH \
       --num-attention-heads $NATT \
       --seq-length $MAXSEQLEN \
       --max-position-embeddings 1024 \
       --fp16 \
       --cache-dir cache \
       --eval-data-path $EVALDATA_PATH \
       --tokenizer-path bpe_3w_new/ \
       --vocab-size 30000 \
       --task tnews "

$CMD
