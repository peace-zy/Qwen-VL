#-*-coding: utf-8-*-

import sys
import os
import json
import argparse
import random
from functools import partial
from tqdm import tqdm
import itertools

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

def collate_fn(batches, tokenizer):

    image_ids = [_['image_id'] for _ in batches]
    questions = [_['question'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    querys = [_['query'] for _ in batches]

    #input_ids = tokenizer(querys, return_tensors='pt', padding='longest')

    return image_ids, querys, questions, answers


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, test):
        self.test = json.load(open(test))

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        ele = self.test[idx]
        image_id, query, answer = ele['id'], ele["conversations"][0]["value"], ele["conversations"][1]["value"]

        question = query.split("</qwen_vl_img>\n")[1]

        return {
            'image_id': image_id,
            "question": question,
            "answer": answer,
            'query': query
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--infile', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--version', type=int, default=0)
    args = parser.parse_args()

    if not args.output:
        args.output = os.path.join("infer_out", os.path.basename(args.checkpoint))
    os.makedirs(args.output, exist_ok=True)

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, device_map='cuda', trust_remote_code=True).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint,
                                              trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id

    prompt = '<img>{}</img>{} Answer:'

    random.seed(args.seed)
    dataset = VQADataset(
        test=args.infile,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    outputs = []

    model.generation_config = GenerationConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
    model.generation_config.top_p = 0.01

    #for _, (image_ids, input_ids, attention_mask,
    #        questions, answers, querys) in tqdm(enumerate(dataloader)):


    for (image_ids, querys, questions, answers) in tqdm(dataloader):
        responses = model.chat_batch(tokenizer, query=querys, history=None)
        for image_id, question, answer, response in zip(image_ids, questions, answers, responses) :
            out_info = {"image_id": image_id, "caption": response, "question": question, "answer": answer}
            outputs.append(out_info)
    torch.distributed.barrier()
    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))
    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() == 0:
        with open(os.path.join(args.output, "{}_{}_res.json".format(args.seed, args.version)), "w") as f:
            #json.dump(merged_outputs, f, indent=4, ensure_ascii=False)
            json.dump(merged_outputs, f)
            #json.dump(outputs, f)
    torch.distributed.barrier()
