from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
#torch.manual_seed(1234)

#model_path = 'output_qwen_sekuai_des_align_zero2_a800_vit_on_cosine_2048/checkpoint-1000'
model_path = 'output_qwen_sekuai_des_align_zero2_a800_vit_on_cosine_2048_ep1'
print(model_path)

print("load tokenizer")
# 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#tokenizer.padding_side = 'left'

# 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# 默认gpu进行推理，需要约24GB显存
print("load model")
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()

# 可指定不同的生成长度、top_p等相关超参（transformers 4.32.0及以上无需执行此操作）
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
model.generation_config.top_p = 0.01

# 第一轮对话
query = tokenizer.from_list_format([
    {'image': '/mnt/aigc_chubao/liyulong/data/v1227/img_single_svg_2/11000014889168.png'}, # Either a local path or an url
    #{'image': 'sample.jpg'},
    #{'text': '图1是原始户型图，图2是改造后的户型图，居住需求如下\n本案适合一家三口常住，但也可能有临时居住需求的家庭，同样可满足未来的二胎计划和老人常住的场景。设计重点以收纳为主体，在装点空间的同时，迈向实用主义。\n图片2的设计理念'},
    #{'text': '一家三口常住，但也可能有临时居住需求的家庭，同样可满足未来的二胎计划和老人常住的场景。设计重点以收纳为主体，在装点空间的同时，迈向实用主义。帮我生成一下改造户型的设计理念'},
    {'text': "用中文讲解下这个户型"},
])



print(query)
print("chat")
query = [query for i in range(2)]
#response, history = model.chat(tokenizer, query=query, history=None)
response = model.chat_batch(tokenizer, query=query, history=None)
print(response)
