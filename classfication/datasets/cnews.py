import numpy as np
from pathlib import Path

# https://blog.csdn.net/rongsenmeng2835/article/details/107437891


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
    print(len(contents[0]), labels[0])


def run():
    w, ws = read_vocab(r"D:\workspace\data\dl\cnews\cnews.vocab.txt")
    c, cs = read_category()
    process_file(filename=r"D:\workspace\data\dl\cnews\cnews.train.txt",
                 word_to_id=ws,
                 cat_to_id=cs,
                 max_length=600)


if __name__ == "__main__":
    run()


