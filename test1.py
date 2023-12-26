# from transformers import GPT2Tokenizer, GPT2Model,GPT2LMHeadModel
# tokenizer = GPT2Tokenizer.from_pretrained('/ssd3/xiaojingwu/gpt2')
# model = GPT2LMHeadModel.from_pretrained('/ssd3/xiaojingwu/gpt2')
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
# print(output)

from huggingface_hub import snapshot_download

repo_id = "xnli"  # 模型在huggingface上的名称
# local_dir = "/ssd3/xiaojingwu/alpaca-lora/llama2/Llama-2-7b-hf/"  # 本地模型存储的地址
local_dir = "/ssd3/xiaojingwu/module/"  # 本地模型存储的地址

local_dir_use_symlinks = False  # 本地模型使用文件保存，而非blob形式保存
token = "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"  # 在hugging face上生成的 access token

# # 如果需要代理的话
proxies = {
    'http': 'http://agent.baidu.com:8118',
    'https': 'http://agent.baidu.com:8118',
}
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    # local_dir_use_symlinks=local_dir_use_symlinks,
    # token=token
    resume_download=True, # 断点续传
    # proxies=proxies
)


# Load model directly


# import json
# from datasets import load_dataset
# intents_file = "/ssd3/xiaojingwu/alpaca-lora/alpaca_data.json"
# with open(intents_file, "r") as f:
#     data = json.load(f)
# data = load_dataset("json", data_files="alpaca-bitcoin-sentiment-dataset.json")
# print(data[0],data["train"])

# intents_file1 = "/ssd3/xiaojingwu/alpaca-lora/alpaca_data_gpt4.json"
# with open(intents_file1, "r") as f:
#     data = json.load(f)
# data = load_dataset("json", data_files="alpaca-bitcoin-sentiment-dataset.json")
# print(data[0],data["train"])

# intents_file2 = "/ssd3/xiaojingwu/alpaca-lora/alpaca_data_cleaned_archive.json"
# with open(intents_file2, "r") as f:
#     data = json.load(f)
# data = load_dataset("json", data_files="alpaca-bitcoin-sentiment-dataset.json")
# print(data[0],data["train"])

# from rouge import Rouge

# # 假设这是参考摘要
# reference = "Load model directly"

# # 假设这是模型生成的摘要
# generated = "Load model direcstly"

# # 初始化ROUGE计算器
# rouge = Rouge()

# # 计算并打印分数
# scores = rouge.get_scores(generated, reference)

# print(scores)


# model_outs = [
#     "this is a nlp lesson",
#     "my name is liutao",
#     "how are you ?"
# ]
# reference = [
#     "this is a text",
#     "your name is liutao",
#     "what do you like ?"
# ]
# scores = rouge.get_scores(model_outs, reference, avg=True)
# # print(scores)




# from datasets import load_dataset,load_from_disk
# dataset = load_dataset('/ssd3/xiaojingwu/cnn_dailymail/','3.0.0')
# print("dataset0")
# print(dataset)
# dataset.save_to_disk('/ssd3/xiaojingwu/datasets/cnn_dailymail')

# print(1)
# from transformers import GPT2LMHeadModel, GPT2Tokenizer 
# print(3)
# import datasets
# # from datasets import load_from_disk
# print(2)
# raw_inputs = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "I hate this so much!",
# ]
# print(raw_inputs)

# model = GPT2LMHeadModel.from_pretrained('/ssd3/xiaojingwu/gpt2')

# tokenizer = GPT2Tokenizer.from_pretrained('/ssd3/xiaojingwu/gpt2')
# print(raw_inputs)
# inputs = tokenizer(raw_inputs)
# print(inputs)
# data = datasets.load_from_disk("/ssd3/xiaojingwu/alpaca-lora/train_dataset")
# print(data)


# import json
# import matplotlib.pyplot as plt
# x = []
# y = []
# cnt = 0
# with open("training_run1.json",'r') as f:
#     for line in f :
#         data = json.loads(line)
#         y.append(data['eval_loss'])
#         x.append(data['epoch'])
# plt.plot(x,y)       