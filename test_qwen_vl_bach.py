from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids, StopWordsLogitsProcessor

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
import pdb
pdb.set_trace()
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

generation_config = generation_config if generation_config is not None else self.generation_config

assert stream is _SENTINEL, _ERROR_STREAM_IN_CHAT
assert generation_config.chat_format == 'chatml', _ERROR_BAD_CHAT_FORMAT
if history is None:
    history = []
if stop_words_ids is None:
    stop_words_ids = []

max_window_size = kwargs.get('max_window_size', None)
if max_window_size is None:
    max_window_size = generation_config.max_window_size
raw_text, context_tokens = make_context(
    tokenizer,
    query,
    history=history,
    system=system,
    max_window_size=max_window_size,
    chat_format=generation_config.chat_format,
)

stop_words_ids.extend(get_stop_words_ids(
    generation_config.chat_format, tokenizer
))
input_ids = torch.tensor([context_tokens]).to(self.device)
outputs = self.generate(
            input_ids,
            stop_words_ids=stop_words_ids,
            return_dict_in_generate=False,
            generation_config=generation_config,
            **kwargs,
        )

response = decode_tokens(
    outputs[0],
    tokenizer,
    raw_text_len=len(raw_text),
    context_length=len(context_tokens),
    chat_format=generation_config.chat_format,
    verbose=False,
    errors='replace'
)

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
