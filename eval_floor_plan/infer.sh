#!/bin/bash
PYTHON_BIN=/data/nfs-ten9/nfs/zhangyan461/env/pytorch_2.1.0_cu12.1_py3.11_qwen_vl/bin/python
#${PYTHON_BIN} infer_floor_plan.py --checkpoint ../output_qwen_sekuai_des_align_zero2_a800_vit_on_cosine_2048_ep1/checkpoint-4500 --output output_qwen_sekuai_des_align_zero2_a800_vit_on_cosine_2048_ep1_checkpoint-4500 --infile ../../dataset/sekuai_des/qwen_format_test_sekuai_des.json
checkpoint=../output_qwen_sekuai_des_align_zero2_a800_vit_on_cosine_2048_ep1
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-1} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    infer_floor_plan_batch.py \
    --checkpoint $checkpoint \
    --infile ../../dataset/sekuai_des/qwen_format_test_sekuai_des.json \
    --output infer_out/output_qwen_sekuai_des_align_zero2_a800_vit_on_cosine_2048_ep1_checkpoint-4500 \
    --batch-size 2 \
    --num-workers 2
