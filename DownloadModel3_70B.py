from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 模型和缓存目录
model_name = "meta-llama/Llama-2-70b-hf"
api_token = "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"
cache_dir = '/work3/s222399/model'

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_token, cache_dir=cache_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_name, token=api_token, cache_dir=cache_dir, num_labels=4)

# 保存tokenizer和模型到磁盘
tokenizer.save_pretrained('/work3/s222399/model_70B')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.save_pretrained('/work3/s222399/model_70B')
model.resize_token_embeddings(len(tokenizer))