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


def main(cfgs, line):
    vocab_file = cfgs["vocab_file"]
    test_file = cfgs["test_file"]

    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 把输入的一行文字信息转换为对应的数字id, 超过600的截掉，没超过的补0
    inputs = [word_to_id[i] for i in line if i in word_to_id] + [0] * (600 - len(line))
    inputs = inputs[:600]
    inputs = torch.LongTensor(inputs).unsqueeze(dim=0)
    inputs = inputs.to(device)

    vocab_size = len(words)

    x_test, y_test = process_file(filename=test_file,
                                  word_to_id=word_to_id,
                                  cat_to_id=cat_to_id,
                                  max_length=600)

    x_test, y_test = torch.LongTensor(x_test), torch.Tensor(y_test)

    model_name = cfgs["model_name"]

    model = eval(model_name)(num_classes=10, vocab_size=vocab_size)
    weight_dict = torch.load(cfgs["train_model_path"], map_location=torch.device("cpu"))
    model.load_state_dict(weight_dict, strict=True)
    model = model.to(device)

    outputs = model(inputs)
    outputs = F.softmax(outputs, dim=-1)

    _, max_index = torch.max(outputs, dim=-1)
    print(line, categories[max_index.item()])


def run():
    # vocab_file = r'D:\workspace\data\dl\cnews\cnews.vocab.txt'
    # test_file = r"D:\workspace\data\dl\cnews\cnews.test.txt"
    cfgs = {
        "model_name": "TextRNN2",
        "vocab_file": "/home/8TDISK/weihule/data/cnews/cnews.vocab.txt",
        "test_file": "/home/8TDISK/weihule/data/cnews/cnews.test.txt",
        "train_model_path": "/home/8TDISK/weihule/data/nlp_training_data/TextRNN2/TextRNN2-0.9398.pth"
    }
    with open(cfgs["test_file"], 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    for line in lines:
        label, content = line.strip().split("\t")
        main(cfgs, content.replace(" ", ""))


if __name__ == "__main__":
    run()
