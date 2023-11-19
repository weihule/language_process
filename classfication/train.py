import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets import process_file, read_vocab, read_category

BATCH_SIZE = 32
EPOCH = 100
LR = 0.001

def main():
    # 获取文本的类别及其对应id的字典
    categories, cat_to_id = read_category()

    # 获取训练文本中所有出现过的字及其所对应的id
    words, word_to_id = read_vocab(r'D:\workspace\data\dl\cnews\cnews.vocab.txt')

    # 获取字数,5000字
    vocab_size = len(words)

    x_train, y_train = process_file(filename=r"D:\workspace\data\dl\cnews\cnews.train.txt",
                                    word_to_id=word_to_id,
                                    cat_to_id=cat_to_id,
                                    max_length=600)

    x_val, y_val = process_file(filename=r"D:\workspace\data\dl\cnews\cnews.val.txt",
                                word_to_id=word_to_id,
                                cat_to_id=cat_to_id,
                                max_length=600)

    x_test, y_test = process_file(filename=r"D:\workspace\data\dl\cnews\cnews.test.txt",
                                  word_to_id=word_to_id,
                                  cat_to_id=cat_to_id,
                                  max_length=600)

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)


if __name__ == "__main__":
    main()
