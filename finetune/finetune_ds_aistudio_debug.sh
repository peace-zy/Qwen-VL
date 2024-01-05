#!/bin/bash
set -x

#ROOT=/data/nfs-ten9/nfs
#mkdir -p ${ROOT} && mount 10.26.192.4:/disk/vdb4/nfs-ten9 ${ROOT}
export CUDA_DEVICE_MAX_CONNECTIONS=1

#PYTHON_TAR_NAME=pytorch_2.1.0_cu12.1_py3.11_qwen_vl.tar
#cp ${ROOT}/zhangyan461/env/${PYTHON_TAR_NAME} ./
#tar -xf ${PYTHON_TAR_NAME}

DIR=`pwd`
export PATH=/aistudio/workspace/qwen_train/env/pytorch_2.1.0_cu12.1_py3.11_qwen_vl/bin:$PATH
export LD_LIBRARY_PATH=/aistudio/workspace/qwen_train/env/pytorch_2.1.0_cu12.1_py3.11_qwen_vl/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
echo ${PATH}
echo ${LD_LIBRARY_PATH}

#ln -s /mnt/aigc_chubao/zhangyan461/dataset dataset

GPUS_PER_NODE=1
#GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

check()
{
    if [ -e $1 ]; then
        echo "[\033[32m"$1"\033[0m] 文件存在"
        
    else
        echo "[\033[31m"$1"\033[0m] 文件不存在"
        exit 1
    fi

}

#MODEL="${ROOT}/zhangyan461/models/Qwen/Qwen-VL-Chat" #"Qwen/Qwen-VL-Chat"/"Qwen/Qwen-VL" # Set the path if you do not want to load from huggingface directly
MODEL="/aistudio/workspace/qwen_train/models/Qwen/Qwen-VL-Chat"
check ${MODEL}
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/mnt/aigc_chubao/zhangyan461/dataset/vlm/visual_instruction_tuning/llava_instruct/LLaVA-Instruct-150K/qwen_format_llava_v1_5_mix665k.json"
check ${DATA}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
TORCHRUN_BIN="/aistudio/workspace/qwen_train/env/pytorch_2.1.0_cu12.1_py3.11_qwen_vl/bin/torchrun"
check ${TORCHRUN_BIN}
LOG_NAME=`date +"%Y_%m_%d_%H_%M_%S"`
${TORCHRUN_BIN} $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 False \
    --fix_vit True \
    --output_dir output_qwen \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --log_level "debug" \
    --deepspeed finetune/ds_config_zero2.json
    #--deepspeed finetune/ds_config_zero2.json > ${LOG_NAME}".log" 2>&1 


set +x
#sleep 2d