#!/bin/bash

#        --load ${CHECKPOINT_PATH} \
       
# bash scripts/chid/finetune_chid_small.sh

# 可以运行了，不加载预训练模型

DATA_DIR="data/chid/"
CHECKPOINT_PATH="../CPM-distill"
RESULTS_DIR="results"
MODEL_NAME="finetune-chid-small"
TOKENIZER_PATH="bpe_3w_new/"

MPSIZE=3
NLAYERS=12
NHIDDEN=768
NATT=12
MAXSEQLEN=1024

CUR_PATH=$(realpath $0)
CUR_DIR=$(dirname ${CUR_PATH})
DS_CONFIG="${CUR_DIR}/../ds_config/ds_finetune_small.json"

python3 -m torch.distributed.launch --master_port ${1-1122} --nproc_per_node ${MPSIZE} finetune_chid.py \
       --do_train \
       --do_eval \
       --data_dir ${DATA_DIR} \
       --model-parallel-size ${MPSIZE} \
       --num-layers ${NLAYERS} \
       --hidden-size ${NHIDDEN} \
       --fp16 \
       --num-attention-heads ${NATT} \
       --seq-length ${MAXSEQLEN} \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --out-seq-length 512 \
       --tokenizer-path ${TOKENIZER_PATH} \
       --vocab-size 30000 \
       --lr 0.00001 \
       --warmup 0.1 \
       --batch-size 4 \
       --deepspeed \
       --deepspeed_config ${DS_CONFIG} \
       --log-interval 10 \
       --eval-interval 1000 \
       --seed 23333 \
       --results_dir ${RESULTS_DIR} \
       --model_name ${MODEL_NAME} \
       --epoch 10 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing
