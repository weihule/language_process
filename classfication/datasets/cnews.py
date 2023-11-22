import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


# https://blog.csdn.net/rongsenmeng2835/article/details/107437891

__all__ = [
    "read_vocab",
    "read_category",
    "process_file"
]


# 读取词汇表, 将其转化为ID
def read_vocab(vocab_dir):
    with open(vocab_dir, "r", encoding="utf-8") as fr:
        lines = fr.readlines()
    words = [line.strip() for line in lines]
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id


# 读取分类目录
def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


# 将文件转为id表示
def process_file(filename, word_to_id, cat_to_id, max_length=600):
    contents, labels = [], []
    with open(filename, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    for line in lines:
        try:
            label, content = line.strip().split("\t")
            if content:
                # 构建双重列表contents,及列表labels
                contents.append(list(content))
                labels.append(label)
        except Exception as e:
            pass
    data_id, label_id = [], []
    for i in range(len(contents)):
        # 将每句话id化
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        # 每句话对应的id添加至对应列表
        label_id.append(cat_to_id[labels[i]])

    data_tensors = [torch.tensor(row) for row in data_id]
    label_tensor = torch.tensor(label_id)
    padded_data = pad_sequence(data_tensors,
                               batch_first=True,
                               padding_value=0.0)
    padded_data = padded_data[:, :600]

    return padded_data, label_tensor


def run():
    w, ws = read_vocab(r"D:\workspace\data\dl\cnews\cnews.vocab.txt")
    c, cs = read_category()
    process_file(filename=r"D:\workspace\data\dl\cnews\cnews.train.txt",
                 word_to_id=ws,
                 cat_to_id=cs,
                 max_length=600)


if __name__ == "__main__":
    run()
