import time
from tqdm import tqdm
from pathlib import Path
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from datasets import process_file, read_vocab, read_category
from models import TextRNN, TextRNN2
from utils.util import mkdirs


def main(cfgs):
    vocab_file = cfgs["vocab_file"]
    test_file = cfgs["test_file"]

    # 获取文本的类别及其对应id的字典
    categories, cat_to_id = read_category()

    # 获取训练文本中所有出现过的字及其所对应的id
    words, word_to_id = read_vocab(vocab_file)

    x_test, y_test = process_file(filename=test_file,
                                  word_to_id=word_to_id,
                                  cat_to_id=cat_to_id,
                                  max_length=600)

    x_test, y_test = torch.LongTensor(x_test), torch.Tensor(y_test)


def run():
    # vocab_file = r'D:\workspace\data\dl\cnews\cnews.vocab.txt'
    # test_file = r"D:\workspace\data\dl\cnews\cnews.test.txt"
    cfgs = {
        "model_name": "TextRNN2",
        "vocab_file": "/home/8TDISK/weihule/data/cnews/cnews.vocab.txt",
        "test_file": "/home/8TDISK/weihule/data/cnews/cnews.test.txt"
    }
    main(cfgs)


if __name__ == "__main__":
    run()
