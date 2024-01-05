from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

model_path = '/mnt/aigc_chubao/zhangyan461/models/Qwen/Qwen-VL'
model_path = '/aistudio/workspace/models/Qwen/Qwen-VL'
model_path = '/aistudio/workspace/qwen_train/Qwen-VL/output_qwen_new_llava_huxinggaizao'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cpu", trust_remote_code=True).eval()
# 默认gpu进行推理，需要约24GB显存
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()

# 可指定不同的生成长度、top_p等相关超参（transformers 4.32.0及以上无需执行此操作）
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
"""
query = tokenizer.from_list_format([
    {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'}, # Either a local path or an url
    {'text': 'Generate the caption in English with grounding:'},
    #{'text': 'Generate the caption in Chinese:'},
])
"""

query = tokenizer.from_list_format([
    {'image': 'dataset/vlm/visual_instruction_tuning/change/before/2f644399c436c1481733bb667f00ca95.jpg'}, # Either a local path or an url
    {'image': 'dataset/vlm/visual_instruction_tuning/change/after/83b07905ac722f9f49eb7071b9a05f53.jpg'},
    #{'text': '图1是原始户型图，图2是改造后的户型图，居住需求如下\n本案适合一家三口常住，但也可能有临时居住需求的家庭，同样可满足未来的二胎计划和老人常住的场景。设计重点以收纳为主体，在装点空间的同时，迈向实用主义。\n图片2的设计理念'},
    {'text': '图1是原始户型图，图2是改造后的户型图，居住需求如下\n本案适合一家三口常住，但也可能有临时居住需求的家庭，同样可满足未来的二胎计划和老人常住的场景。设计重点以收纳为主体，在装点空间的同时，迈向实用主义。\n帮我生成一下改造户型的设计理念'},
])
"""
query = tokenizer.from_list_format([
    {'image': 'dataset/vlm/visual_instruction_tuning/change/before/2f644399c436c1481733bb667f00ca95.jpg'}, # Either a local path or an url
    {'text': 'Generate the caption in Chinese:'},
])
"""

inputs = tokenizer(query, return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
print("dddd")
print("response={}".format(response[len(query)+1:]))
# <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>Generate the caption in English with grounding:<ref> Woman</ref><box>(451,379),(731,806)</box> and<ref> her dog</ref><box>(219,424),(576,896)</box> playing on the beach<|endoftext|>
image = tokenizer.draw_bbox_on_latest_picture(response)
if image:
  image.save('3.jpg')
else:
  print("no box")
