from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
#torch.manual_seed(1234)

model_path = '/mnt/aigc_chubao/zhangyan461/models/Qwen/Qwen-VL-Chat'
model_path = '/data/nfs-ten9/nfs/zhangyan461/models/Qwen/Qwen-VL-Chat'
model_path = '/aistudio/workspace/qwen_train/Qwen-VL/output_qwen_new_llava_huxingjiedu'
model_path = '/aistudio/workspace/models/Qwen/Qwen-VL-Chat'



model_path = '/aistudio/workspace/qwen_train/Qwen-VL/output_qwen_new_llava_huxinggaizao'
model_path = '/aistudio/workspace/qwen_train/Qwen-VL/output_qwen_new_llava_huxinggaizao_zero3'

model_path = '/aistudio/workspace/qwen_train/Qwen-VL/output_qwen_new_llava_huxinggaizao_zero3_a100'
model_path = '/aistudio/workspace/qwen_train/models/Qwen/Qwen-VL-Chat'
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
"""
query = tokenizer.from_list_format([
    {'image': 'dataset/vlm/visual_instruction_tuning/house_article/second_hand_house/d27bad8411b0548d849b27ac5defc289.jpg'}, # Either a local path or an url
    {'text': '以设计师的口吻，用中文描述下这张图'},
    #{'text': '用中文讲解下这个户型图'},
])

query = tokenizer.from_list_format([
    {'image': 'dataset/vlm/visual_instruction_tuning/house_article/second_hand_house/d35006c20b4319928c93379fa68def35.jpg'}, # Either a local path or an url
    {'text': '这是什么?'},
])
query = tokenizer.from_list_format([
    {'image': 'dataset/vlm/visual_instruction_tuning/change/before/2f644399c436c1481733bb667f00ca95.jpg'}, # Either a local path or an url
    {'image': 'dataset/vlm/visual_instruction_tuning/change/after/83b07905ac722f9f49eb7071b9a05f53.jpg'},
    {'text': '图1是原始户型图，图2是改造后的户型图，居住需求如下\n本案适合一家三口常住，但也可能有临时居住需求的家庭，同样可满足未来的二胎计划和老人常住的场景。设计重点以收纳为主体，在装点空间的同时，迈向实用主义。\n图片2的设计理念'},
    #{'text': '图1是原始户型图，图2是改造后的户型图，居住需求如下\n本案适合一家三口常住，但也可能有临时居住需求的家庭，同样可满足未来的二胎计划和老人常住的场景。设计重点以收纳为主体，在装点空间的同时，迈向实用主义。\n帮我生成一下改造户型的设计理念'},
])

query = tokenizer.from_list_format([
    #{'image': 'dataset/vlm/visual_instruction_tuning/change/before/2f644399c436c1481733bb667f00ca95.jpg'}, # Either a local path or an url
    {'image': 'dataset/vlm/visual_instruction_tuning/change/after/83b07905ac722f9f49eb7071b9a05f53.jpg'},
    #{'text': '图1是原始户型图，图2是改造后的户型图，居住需求如下\n本案适合一家三口常住，但也可能有临时居住需求的家庭，同样可满足未来的二胎计划和老人常住的场景。设计重点以收纳为主体，在装点空间的同时，迈向实用主义。\n图片2的设计理念'},
    #{'text': '一家三口常住，但也可能有临时居住需求的家庭，同样可满足未来的二胎计划和老人常住的场景。设计重点以收纳为主体，在装点空间的同时，迈向实用主义。帮我生成一下改造户型的设计理念'},
    {'text': '帮我描述一下这个户型的设计理念'},
])

query = tokenizer.from_list_format([
    #{'image': 'dataset/vlm/visual_instruction_tuning/change/before/2f644399c436c1481733bb667f00ca95.jpg'}, # Either a local path or an url
    {'image': '/aistudio/workspace/qwen_train/Qwen-VL/d3a8695ec3073e196f55e7005fe3c118c7bee0df-4096-3115.jpeg'},
    #{'text': '图1是原始户型图，图2是改造后的户型图，居住需求如下\n本案适合一家三口常住，但也可能有临时居住需求的家庭，同样可满足未来的二胎计划和老人常住的场景。设计重点以收纳为主体，在装点空间的同时，迈向实用主义。\n图片2的设计理念'},
    #{'text': '一家三口常住，但也可能有临时居住需求的家庭，同样可满足未来的二胎计划和老人常住的场景。设计重点以收纳为主体，在装点空间的同时，迈向实用主义。帮我生成一下改造户型的设计理念'},
    {'text': '帮我描述一下这个户型的设计理念'},
])

query = tokenizer.from_list_format([
    #{'image': 'dataset/vlm/visual_instruction_tuning/change/before/2f644399c436c1481733bb667f00ca95.jpg'}, # Either a local path or an url
    {'image': 'dataset/vlm/visual_instruction_tuning/change/before/606d58a13a334e730bd980e4dde6e5af.jpg'},
    {'image': 'dataset/vlm/visual_instruction_tuning/change/after/652e0ed0f5776b03a42e9cf83d216971.jpg'},
    #{'text': '图1是原始户型图，图2是改造后的户型图，居住需求如下\n本案适合一家三口常住，但也可能有临时居住需求的家庭，同样可满足未来的二胎计划和老人常住的场景。设计重点以收纳为主体，在装点空间的同时，迈向实用主义。\n图片2的设计理念'},
    #{'text': '一家三口常住，但也可能有临时居住需求的家庭，同样可满足未来的二胎计划和老人常住的场景。设计重点以收纳为主体，在装点空间的同时，迈向实用主义。帮我生成一下改造户型的设计理念'},
    {'text': '图1是原始户型图，图2是改造后的户型图，居住需求如下\n本案适合一家三口常住的家庭，尤其是孩子正处于上学阶段，拥有艺术爱好的情况，次卧设置兴趣学习区，书桌的设计更有童趣。床下方也可用来收纳书籍和储藏衣物；同时卧室榻榻米背靠吧台，打造书吧咖啡厅的主题，更有氛围，也让加班时间充满阳光和希望。\n帮我生成一下改造户型的设计理念'},
])

query = tokenizer.from_list_format([
    #{'image': 'dataset/vlm/visual_instruction_tuning/change/before/2f644399c436c1481733bb667f00ca95.jpg'}, # Either a local path or an url
    {'image': 'sample.jpg'},
    {'image': 'dataset/vlm/visual_instruction_tuning/change/after/652e0ed0f5776b03a42e9cf83d216971.jpg'},
    #{'text': '图1是原始户型图，图2是改造后的户型图，居住需求如下\n本案适合一家三口常住，但也可能有临时居住需求的家庭，同样可满足未来的二胎计划和老人常住的场景。设计重点以收纳为主体，在装点空间的同时，迈向实用主义。\n图片2的设计理念'},
    #{'text': '一家三口常住，但也可能有临时居住需求的家庭，同样可满足未来的二胎计划和老人常住的场景。设计重点以收纳为主体，在装点空间的同时，迈向实用主义。帮我生成一下改造户型的设计理念'},
    {'text': '图1户型图的缺点主卧、次卧、卫浴，门洞都朝客厅开，客厅门太多，影响空间利用率，视觉美感差；次卧采光差；厨房空间狭小；卫浴间空间狭小\n帮我描述一下第二图的户型缺点'},
])
query = tokenizer.from_list_format([
    #{'image': 'dataset/vlm/visual_instruction_tuning/change/before/2f644399c436c1481733bb667f00ca95.jpg'}, # Either a local path or an url
    {'image': 'sample.jpg'},
    #{'text': '图1是原始户型图，图2是改造后的户型图，居住需求如下\n本案适合一家三口常住，但也可能有临时居住需求的家庭，同样可满足未来的二胎计划和老人常住的场景。设计重点以收纳为主体，在装点空间的同时，迈向实用主义。\n图片2的设计理念'},
    #{'text': '一家三口常住，但也可能有临时居住需求的家庭，同样可满足未来的二胎计划和老人常住的场景。设计重点以收纳为主体，在装点空间的同时，迈向实用主义。帮我生成一下改造户型的设计理念'},
    {'text': '该户型图的缺点主卧、次卧、卫浴，门洞都朝客厅开，客厅门太多，影响空间利用率，视觉美感差；次卧采光差；厨房空间狭小；卫浴间空间狭小'},
])
"""
a ="""以设计师的角度，针对下面的文本生成客户的居住需求，户型改造前的优缺点，户型改造点和户型改造理念，输入信息如下：
{"materials":{"m_0":{"data":"
户型
一居
风格
欧式风
面积
55㎡
位置
北京市
方式
全包
预算
8万
","type":"html"},"m_1":{"data":"
今天的这套房子是位于北京朝阳区的一套老破小。\n\nQ先生买的这套建面积55㎡的学区房，完全符合“老破小”。小区的配套设施与周边还是非常不错的。\n\n接下来我们一起看看这期的“主角”，如何“老炮”换“新颜”！



原始问题：\n\n● 原始户型中，卫生间是不足2平方，洗澡都成问题；\n\n● 房子位置在北京三环公路旁，车水马龙的噪音；\n\n● 小小的厨房，容纳不下很多常用电器，家电的选择受限；\n\n● 老破小的诟病，储藏储物空间需要合理规划与设计；\n\n● 门厅玄关狭小没有足够的收纳空间；\n\n● 阳台没有下水口，无法将洗衣机安放其中。



对于生活品质有一定要求的Q先生，生活需求与装修需求，尽可能都在装修设计中实现了。\n\n改造方案：\n\n● 拆改原始户型布局，增加卫生间可用面积，实现三分离；\n\n● 公共空间精细的规划后，调整空间功能属性，将餐厅和办公区完美实现；\n\n● 主卧与客厅空间都增加通顶储物柜，满足收纳需求。\n\n● 冰箱尽可能要在厨房里，让客厅不拥挤。\n\n● 阳台作为男主的办公区，安静惬意，阳光充足；



","type":"html"},"m_2":{"data":"


入户的玄关，进入眼帘的灰色水泥墙面做为空间主调，狭小的玄关空间没有用通顶柜的设计，而是采用灵活、储藏功能全面的小型收纳架做替代，让原本压抑的空间多了一点呼吸。





入户门右侧的墙面上，安装了一个多功能衣架，放一下外套、小背包、健身包都可以！\n\n防止未来的日子挂的物件会把墙面划伤或者染上颜色，内部也做了相同的水泥墙面（刷的水泥漆），也形成小小的呼应。





","type":"html"},"m_3":{"data":"


原始的空间为卧室，设计师结合业主需求更改了空间功能属性。\n\n改造前



改造后



将邻近窗户的位置，作为餐厅区，用餐的同时不忘欣赏窗外的景色（楼层位置很好，窗外并没有被大树所遮挡）



好看的白色玻璃柜中放满各种精致的玻璃酒杯，期待有朋自远方来，把酒言欢一醉方休的情景。



把酒言欢也少不了下酒菜，餐边柜中陈列着好看且精致的器皿，更多是用来艺术欣赏。







客厅的储物空间也是必须要有的，好看的纯白通顶大衣柜与空间更协调。



衣柜内部，准备了一些挂衣区，一来挂起来的衣服一目了然，找衣服方便，再就是衣服叠放容易起褶皱，挂起来也可以避免这个问题。\n\n板材上有均匀的孔洞，可以后期根据不同的需要来调节内部结构。

","type":"html"},"m_4":{"data":"


不足4㎡的厨房，真的是老破小的诟病之一。老话说的好，麻雀虽小但五脏俱全啊~



U型橱柜也将洗切炒的区域，给拿捏死死的，灶具的位置是向左侧移动了将近10公分，台面的空间就节省了一部分，这样冰箱也在厨房有了安身之处。\n\n肯定看到这里啊，你们会有想问:“为啥这么小的空间将冰箱放进来呢？不会放到其他空间吗？”\n\n其实我们的屋主朋友对于噪音是比较在意的，放在厨房也是最佳方案，如果对冰箱放置位置有不同思路的小伙伴儿在留言区留言我们一起来聊聊！



","type":"html"},"m_5":{"data":"


拆掉原有的窗体，让阳台与室内融合在一起，通透感更强，室内采光更佳，视野中横向深度更好空间感觉更大更宽敞一些。\n\n改造前





改造前，阳台还保留着80年90年代的封闭式阳台的样式，空间的呼吸感通透感不足，没有将采光完好的引入室内。\n\n改造后



改造后，拆除了窗体，在阳台和卧室之间，用大地色窗帘作为软性隔断，有效分割了空间，巧妙的设计独具风格。\n\n为了满足办公需求，将采光很好的阳台空间改造成办公区，阳光充足，相对安静的阳台，完美符合Q先生的需求。





主卧较为柔和的设计，背景墙没有过多的造型，主要是以浅灰色的墙色+功能吊灯加以装饰，显得简洁大气。



一组轨道灯凸显出现代极简风，柔和的，散发着光晕的灯光给简约的卧室空间增添了些许生活的气息。



","type":"html"},"m_6":{"data":"


原始户型中，淋浴与马桶相拥在不足2平方的小空里，非常不方便。



改造后，淋浴区、马桶间与洗漱区做到了三分离，各个空间面积得到提升。



不足2平方的空间，放置一个常规的智能马桶，马桶旁放置了一个20公分的多层储物架，一些生活物品有了放置平台。\n\n悬空而做的平台上放置一些艺术气息十足的饰品与香薰，平台下方装有感应灯带，让小小的空间不再压抑。



可爱的小黄鸭，在我们洗漱区的最中心的位置~\n\n原本空间不充裕的洗漱区，要装下一个洗衣机和安装一个合适高度的水盆。何尝不是一种挑战！



超薄洗衣机是此空间中的首选，一体式台下盆洗漱台将空间做好了规划。\n\n细心的Q先生逛家居生活商场时，巧合的发现尺寸完美的衣物收纳篮与小型浴室柜。



原本属于储物的小空间，改造成了淋浴区。



小户型中，能做到三分离的非常不容易，但是不足1.5平方的淋浴区，浴室门开合所扫过的空间也是在日常使用中要腾出空间的。\n\n如果可以将平开门更换成折叠门是不是会更好一点呢~（留言区可以说一说您的看法）



","type":"html"},"m_7":{"data":"
更好的生活源于对生活的热爱，一种敬重和热爱生活态度，对舒适与热情的追求。\n\n最后分享此案例中的生活艺术角，也希望我们自己也能带着发现美的眼睛，去探寻就在身边的美！"""

query = tokenizer.from_list_format([
    #{'image': 'dataset/vlm/visual_instruction_tuning/change/before/2f644399c436c1481733bb667f00ca95.jpg'}, # Either a local path or an url
    #{'image': 'sample.jpg'},
    #{'text': '图1是原始户型图，图2是改造后的户型图，居住需求如下\n本案适合一家三口常住，但也可能有临时居住需求的家庭，同样可满足未来的二胎计划和老人常住的场景。设计重点以收纳为主体，在装点空间的同时，迈向实用主义。\n图片2的设计理念'},
    #{'text': '一家三口常住，但也可能有临时居住需求的家庭，同样可满足未来的二胎计划和老人常住的场景。设计重点以收纳为主体，在装点空间的同时，迈向实用主义。帮我生成一下改造户型的设计理念'},
    {'text': a},
])



print(query)
print("chat")
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
