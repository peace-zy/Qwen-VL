import argparse
import itertools
import json
import os
import random
import time
import jieba
from tqdm import tqdm
from functools import partial

from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO


gt_info = {
    'flickr': {
        'train': 'data/flickr30k/flickr30k_karpathy_test.json',
        'test': 'data/flickr30k/flickr30k_karpathy_test.json',
    },
    'nocaps': {
        'train': '',
        'test': 'data/nocaps/nocaps_val.json',
    },
    'huxinggaizao': {
        'train': '',
        'test': '/aistudio/workspace/qwen_train/dataset/huxinggaizao_test_ori.json',
    }
    
}

def gen_standard_data_and_process_with_jieba(gt_file, res_file, args, ):
    with open(gt_file, "r") as f:
        data = json.load(f)
    d = set([])
    for ele in data:
        if ele["id"] in d:
            print(ele["id"])
        else:
            d.add(ele["id"])
    # gt standardization
    standard_gt_out = {"annotations": [], "images": []}
    idx = 0
    for ele in tqdm(data):
        image_id, query, answer = ele['id'], ele["conversations"][0]["value"], ele["conversations"][1]["value"]
        #phrase-wise
        if args.cut_mode == "phrase":
            words = jieba.cut(answer.strip(), cut_all=False)
            answer = " ".join(words)
        elif args.cut_mode == "word":
            #word-wise
            answer = " ".join(answer)
        standard_gt_out["annotations"].append({"image_id": image_id,
                                   "id": 0,
                                   "image": image_id,
                                   "caption": answer})
        standard_gt_out["images"].append({"id": image_id,
                              "image": image_id})
        idx += 1
    out_dataset_path = os.path.join(args.save_path, args.dataset)
    out_path = os.path.join(out_dataset_path, str(args.version))
    os.makedirs(out_path, exist_ok=True)
    new_gt_file = os.path.join(out_dataset_path, "{}_gt.txt".format(args.dataset))
    with open(new_gt_file, "w") as f:
        json.dump(standard_gt_out, f, indent=4, ensure_ascii=False)

    new_res_file = os.path.join(out_path, "{}_{}_res.txt".format(args.dataset, args.version))
    if args.check:
        # verify correct
        with open(new_gt_file, "r") as f:
            data = json.load(f)
        out = []
        for ele in data["annotations"]:
            out.append({"image_id": ele["image_id"],
                        "caption": ele["caption"]})
        with open(new_res_file, "w") as f:
            json.dump(out, f, indent=4, ensure_ascii=False)
    else:
        # res standardization
        with open(res_file, "r") as f:
            data = json.load(f)
        
        for ele in tqdm(data):
            if args.cut_mode == "phrase":
                words = jieba.cut(ele["caption"].strip(), cut_all=False)
                ele["caption"] = " ".join(words)
            elif args.cut_mode == "word":
                #word-wise
                ele["caption"] = " ".join(ele["caption"])
            pass
    
    
        with open(new_res_file, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    return (new_gt_file, new_res_file)
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='', help="dataset")
    parser.add_argument('--standardization', action="store_true", help="use with cut_word or cut_phrase for chinese caption task")
    parser.add_argument('--cut_mode', type=str, default="word", help="[word or phrase or none] level")
    parser.add_argument('--check', action="store_true", help="use train data to check correctness")
    parser.add_argument('--save_path', type=str, default='eval_out', help="save path")
    parser.add_argument('--version', type=str, default="1", help="version")
    parser.add_argument('--submit', action="store_true", help="submit speified gt_file and res_file")
    parser.add_argument('--gt_file', type=str, default='', help="specified gt_file")
    parser.add_argument('--res_file', type=str, default='', help="specified res_file")
    args = parser.parse_args()

    dataset = args.dataset
    res_files = {"nocaps": ['nocaps_231122045203.json', 'nocaps_231122063036.json'],
                 "flickr": ['flickr_231123182439.json'],
                 "huxinggaizao": ["/aistudio/workspace/qwen_train/Qwen-VL/Qwen-VL-Chat-huxinggaizao_office/res.json"]}
                 #"huxinggaizao": ["/aistudio/workspace/qwen_train/Qwen-VL/Qwen-VL-Chat-huxinggaizao_d/res.json"]}
    if args.submit:
        if args.gt_file:
            if dataset not in gt_info:
                gt_info[dataset] = {}
            gt_info[dataset]['test'] = args.gt_file
        if args.res_file:
            if dataset not in res_files:
                res_files[dataset] = []
            res_files[dataset] = [args.res_file]

    for results_file in res_files[dataset]:
        gt_file = gt_info[dataset]['test']
        if args.standardization:
            gt_file, results_file = gen_standard_data_and_process_with_jieba(gt_file, results_file, args)
        coco = COCO(gt_file)
        coco_result = coco.loadRes(results_file)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.evaluate(spice=False)

        # create output dictionary
        out = {}
        for metric, score in coco_eval.eval.items():
            out[metric] = "{:.3f}".format(score)
        imgToEval = coco_eval.imgToEval
        for x in imgToEval:
            imgToEval[x]['caption'] = coco_result.imgToAnns[x]

        out_dataset_path = os.path.join(args.save_path, args.dataset)
        out_path = os.path.join(args.save_path, args.dataset, str(args.version))
        os.makedirs(out_path, exist_ok=True)

        with open(os.path.join(out_path, "{}_{}_{}_performance.txt".format(args.dataset, args.cut_mode, args.version)), 'w') as outfile:
            json.dump({'overall': out, 'imgToEval': imgToEval}, outfile, indent=4, ensure_ascii=False)


        print("\033[32m[{}]\033[0m {}".format(results_file, coco_eval.eval.items()))
