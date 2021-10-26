# 环境：
## cuda
cuda 11.0，pytorch 1.8.0
## apex
应该用cuda 10.2，cuda 11.0 不兼容，参考[这里](https://zhuanlan.zhihu.com/p/80386137)修改apex，成功编译
## deepspeed
pip install deepspeed==0.3.15

# 任务
## 1. zero-shot classication
dataset: tnews
train: scripts/zero-shot-tnews_small.sh
## 2. fill-in-the-blank
dataset: chid
dataprocess: preprocess_chid_finetune.py
train: scripts/chid/finetune_chid_small.sh
## 3. dialog
dataset: STC
dataprocess: preprocess_stc_finetune.py
train: finetune_lm_small.sh

# 待改进：
1. 预训练模型加载不成功
2. 大模型显存占用过大
