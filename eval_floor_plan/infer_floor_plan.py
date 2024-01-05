#-*-coding: utf-8-*-

import sys
import os
import json
import argparse
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

"""
version = sys.version_info.major
if version >= 3:
    import importlib
    importlib.reload(sys)
elif version >= 2:
    reload(sys)
    sys.setdefaultencoding('utf8')
print(sys.getdefaultencoding())

#checkpoint = '/aistudio/workspace/qwen_train/Qwen-VL/output_qwen_new_llava_huxinggaizao'
checkpoint = '/aistudio/workspace/qwen_train/models/Qwen/Qwen-VL-Chat'
checkpoint = '/aistudio/workspace/qwen_train/Qwen-VL/output_qwen_new_llava_huxinggaizao_zero3'

#checkpoint = '/mnt/aigc_chubao/zhangyan461/models/Qwen/Qwen-VL-Chat'
"""


"""
output = 'Qwen-VL-Chat-huxinggaizao_d'
output = 'Qwen-VL-Chat-huxinggaizao_office'
output = 'Qwen-VL-Chat-huxinggaizao_zero3'
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--infile', type=str, default='')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, device_map='cuda', trust_remote_code=True).eval()

    model.generation_config = GenerationConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
    model.generation_config.top_p = 0.01


    if not args.output:
        args.output = os.path.join("infer_out", os.path.basename(args.checkpoint))
        
    os.makedirs(args.output, exist_ok=True)
    with open(args.infile, "r") as f:
        data = json.load(f)
    out = []
    for ele in tqdm(data):
        image_id, query, answer = ele['id'], ele["conversations"][0]["value"], ele["conversations"][1]["value"]
        #question = query.split("居住需求如下\n")[1].split("\n帮我生成一下改造户型的设计理念")[0]
        question = query.split("</qwen_vl_img>\n")[1]
        #print("question={}".format(question))
        response, _ = model.chat(tokenizer, query=query, history=None)
        #print("response={}".format(response))
        out.append({"image_id": image_id, "caption": response, "question": question, "answer": answer})
    with open(os.path.join(args.output, "res.json"), "w") as f:
        #json.dump(out, f, indent=4, ensure_ascii=False)
        json.dump(out, f)
