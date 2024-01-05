#!/bin/bash
#../env/pytorch_2.1.0_cu12.1_py3.11_qwen_vl/bin/python eval_mm/eval_huxinggaizao.py --infile ../dataset/huxinggaizao_test_ori.json --checkpoint output_qwen_new_llava_huxinggaizao_zero3 --output Qwen-VL-Chat-huxinggaizao_zero3 > Qwen-VL-Chat-huxinggaizao_zero3.log
#/aistudio/workspace/system-default/envs/pytorch_2.1.0_cu12.1_py3.11_qwen_vl/bin/python eval_floor_plan/infer_floor_plan.py --infile ../dataset/huxinggaizao_test_ori.json --checkpoint /aistudio/workspace/qwen_train/Qwen-VL/output_qwen_new_llava_huxinggaizao_zero3_a100 --output eval_floor_plan/Qwen-VL-Chat-huxinggaizao_zero3_cosine
#/aistudio/workspace/system-default/envs/pytorch_2.1.0_cu12.1_py3.11_qwen_vl/bin/python eval_floor_plan/infer_floor_plan.py --infile ../dataset/sekuai_des/qwen_format_test_sekuai_des.json --checkpoint /aistudio/workspace/qwen_train/Qwen-VL/output_qwen_sekuai_des_align_zero2_a800_vit_on_cosine_2048_ep1 --output eval_floor_plan/Qwen-VL-Chat-sekuai_des_align_zero2_a800_vit_on_cosine_2048_ep1
checkpoint=./output_qwen_sekuai_des_align_zero2_a800_vit_on_cosine_2048_ep1
checkpoint=./output_qwen_sekuai_des_align_zero2_a800_vit_on_lm_off_cosine_2048_ep1
/aistudio/workspace/system-default/envs/pytorch_2.1.0_cu12.1_py3.11_qwen_vl/bin/python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-2} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    eval_floor_plan/infer_floor_plan_batch_chat.py \
    --checkpoint $checkpoint \
    --infile ../dataset/sekuai_des/qwen_format_test_sekuai_des.json \
    --output eval_floor_plan/infer_out/output_qwen_sekuai_des_align_zero2_a800_vit_on_lm_off_cosine_2048_ep1_checkpoint-4500_chat \
    --batch-size 20 \
    --num-workers 2
#eval_floor_plan/infer_floor_plan_batch.py \
