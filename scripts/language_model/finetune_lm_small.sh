#!/bin/bash

# bash scripts/language_model/finetune_lm_small.sh

       # --load ${CHECKPOINT_PATH} \
# 
# 这个跑通了，不加载预训练模型

DATA_DIR="data/STC"
CHECKPOINT_PATH="../CPM-distill"
RESULTS_DIR="results/"
MODEL_NAME="finetune-dial-small"
TOKENIZER_PATH="bpe_3w_new/"

MPSIZE=3
NLAYERS=12
NHIDDEN=768
NATT=12
MAXSEQLEN=200

CUR_PATH=$(realpath $0)
CUR_DIR=$(dirname ${CUR_PATH})
DS_CONFIG="${CUR_DIR}/../ds_config/ds_finetune_lm_large.json"

python3 -m torch.distributed.launch --master_port ${1-1122} --nproc_per_node ${MPSIZE} finetune_dialog.py \
       --do_train \
       --do_eval \
       --data_dir ${DATA_DIR} \
       --model-parallel-size ${MPSIZE} \
       --num-layers ${NLAYERS} \
       --hidden-size ${NHIDDEN} \
       --num-attention-heads ${NATT} \
       --seq-length ${MAXSEQLEN} \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --tokenizer-path ${TOKENIZER_PATH} \
       --vocab-size 30000 \
       --lr 0.00001 \
       --warmup 0.1 \
       --batch-size 16 \
       --deepspeed \
       --deepspeed_config ${DS_CONFIG} \
       --log-interval 10 \
       --eval-interval 100 \
       --seed 23333 \
       --results_dir ${RESULTS_DIR} \
       --model_name ${MODEL_NAME} \
       --epoch 1 \
       --checkpoint-activations \
       # --deepspeed-activation-checkpointing
