import torch
import gc
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 加载数据
df = pd.read_csv('./sentence_data.csv')

# 准备数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 加载保存的tokenizer和模型
# model_name = "meta-llama/Llama-2-13b-hf"
# api_token = "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"
# cache_dir = '/path/to/large_disk_space'
# tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_token, cache_dir=cache_dir)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, token=api_token, cache_dir=cache_dir, num_labels=4)
tokenizer = AutoTokenizer.from_pretrained('/work3/s222399/model')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForSequenceClassification.from_pretrained('/work3/s222399/model')
model.resize_token_embeddings(len(tokenizer))

# 编码数据集的函数
def encode_data(df):
    encodings = tokenizer(list(df['sentence']), truncation=True, padding=True, max_length=512)
    labels = df[['T1', 'T2', 'T3', 'T4']].apply(lambda x: [1 if val.endswith('_true') else 0 for val in x], axis=1).tolist()
    return encodings, labels

# 初始化KFold
kf = KFold(n_splits=5)

# 训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,  # 增加批次大小
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # 累积梯度步数
    fp16=True,  # 使用混合精度
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch"
)

# 交叉验证
for train_index, val_index in kf.split(df):
    train_df, val_df = df.iloc[train_index], df.iloc[val_index]

    # 编码数据
    train_encodings, train_labels = encode_data(train_df)
    val_encodings, val_labels = encode_data(val_df)

    # 创建数据集
    train_dataset = Dataset(train_encodings, train_labels)
    val_dataset = Dataset(val_encodings, val_labels)

    # 定义Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # 训练模型
    trainer.train()

    # 清理内存
    torch.cuda.empty_cache()
    gc.collect()

    # 评估模型
    predictions = trainer.predict(val_dataset)
    preds = torch.argmax(torch.tensor(predictions.predictions), axis=1)
    labels = torch.tensor(val_labels)

    print("Classification Report")
    for i, label in enumerate(['T1', 'T2', 'T3', 'T4']):
        print(f"Classification Report for {label}")
        print(classification_report(labels[:, i], preds[:, i], target_names=[f'not_{label}', label]))

    print("Confusion Matrix")
    for i, label in enumerate(['T1', 'T2', 'T3', 'T4']):
        print(f"Confusion Matrix for {label}")
        print(confusion_matrix(labels[:, i], preds[:, i]))
