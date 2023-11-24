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

# r"D:\workspace\data\nlp_training_data"
cfgs = {
    "batch_size": 64,
    "epoch": 100,
    "lr": 0.001,
    "model_name": "TextRNN2",
    "save_root": "/home/8TDISK/weihule/data/nlp_training_data",
    "print_interval": 80,
    "save_interval": 4,
    "mode": "1.65"
}


def train(cfgs, logger, model, train_loader, criterion, optimizer, scheduler, epoch, device):
    # 设置训练模式
    model.train()

    mean_loss = 0.
    iter_idx = 1
    for ds in tqdm(train_loader):
        inputs, labels = ds
        inputs, labels = inputs.to(device), labels.to(device)
        # labels = F.one_hot(labels, num_classes=model.numclasses)
        # 梯度清零
        optimizer.zero_grad()

        preds = model(inputs)
        loss = criterion(preds, labels)
        mean_loss += loss.item()

        # 损失回传
        loss.backward()

        # 梯度更新
        optimizer.step()

        if iter_idx % cfgs["print_interval"] == 0 or iter_idx == len(train_loader):
            lr_value = optimizer.state_dict()['param_groups'][0]['lr']
            logger.info(
                f"train epoch {epoch:3d}, iter [{iter_idx:5d}, {len(train_loader)}], loss: {loss.item():.3f} lr:{lr_value}")
        iter_idx += 1

    mean_loss = mean_loss / len(train_loader)
    scheduler.step()

    return mean_loss


def evaluate(model, val_dataset_len, val_loader, device):
    correct = 0.
    model.eval()
    model.to(device)
    with torch.no_grad():
        for ds in tqdm(val_loader):
            inputs, labels = ds
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=-1)

            _, max_indices = torch.max(outputs, dim=-1)
            correct += torch.eq(labels, max_indices).sum().item()
        val_acc = round(correct / val_dataset_len, 4)

    return val_acc


def init_dir(cfgs):
    save_root = cfgs["save_root"]
    mkdirs(save_root)

    model_dir = Path(save_root) / cfgs["model_name"]
    mkdirs(model_dir)

    return model_dir


# 创建一个自定义的 TextIOWrapper 对象
class TqdmStream:
    def write(self, msg):
        tqdm.write(msg)


def main():
    if cfgs["mode"] == "local":
        vocab_file = r'D:\workspace\data\dl\cnews\cnews.vocab.txt'
        train_file = r"D:\workspace\data\dl\cnews\cnews.train.txt"
        val_file = r"D:\workspace\data\dl\cnews\cnews.val.txt"
    elif cfgs["mode"] == "1.65":
        vocab_file = "/home/8TDISK/weihule/data/cnews/cnews.vocab.txt"
        train_file = "/home/8TDISK/weihule/data/cnews/cnews.train.txt"
        val_file = "/home/8TDISK/weihule/data/cnews/cnews.val.txt"
    else:
        vocab_file = ""
        train_file = ""
        val_file = ""

    model_dir: Path = init_dir(cfgs)

    model_name = cfgs["model_name"]

    # 配置日志输出到文件
    logger.add(model_dir / f"{model_name}.log", rotation="500 MB", level="INFO")

    # 获取文本的类别及其对应id的字典
    categories, cat_to_id = read_category()

    # 获取训练文本中所有出现过的字及其所对应的id
    words, word_to_id = read_vocab(vocab_file)

    # 获取字数,5000字
    vocab_size = len(words)

    x_train, y_train = process_file(filename=train_file,
                                    word_to_id=word_to_id,
                                    cat_to_id=cat_to_id,
                                    max_length=600)

    x_val, y_val = process_file(filename=val_file,
                                word_to_id=word_to_id,
                                cat_to_id=cat_to_id,
                                max_length=600)

    x_train, y_train = torch.LongTensor(x_train), torch.Tensor(y_train)
    x_val, y_val = torch.LongTensor(x_val), torch.Tensor(y_val)

    # 利用 TensorDataset 直接将x_train, y_train整合成Dataset结构
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfgs["batch_size"],
        shuffle=True,
        num_workers=2,
    )

    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfgs["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = eval(model_name)(num_classes=10, vocab_size=vocab_size)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfgs["lr"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=cfgs["epoch"])

    best_acc = 0.
    start_epoch = 1

    start_time = time.time()
    for epoch in range(start_epoch, cfgs["epoch"] + 1):
        mean_loss = train(cfgs=cfgs,
                          logger=logger,
                          model=model,
                          train_loader=train_loader,
                          criterion=criterion,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          epoch=epoch,
                          device=device)
        logger.info(f"train: epoch: {epoch}, loss: {mean_loss:.3f}")
        if epoch % cfgs["save_interval"] == 0 or epoch == cfgs["epoch"]:
            val_acc = evaluate(model=model,
                               val_dataset_len=len(val_dataset),
                               val_loader=val_loader,
                               device=device)
            logger.info(f"epoch = {epoch}, val_acc = {val_acc}")
            if val_acc > best_acc:
                # 先删除旧的历史权重
                for i in model_dir.glob(f"{model_name}*.pth"):
                    i.unlink(missing_ok=True)
                best_acc = val_acc
                best_weight_name = cfgs["model_name"] + "-" + str(best_acc) + ".pth"
                torch.save(model.state_dict(), str(model_dir / best_weight_name))

                torch.save({
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'loss': mean_loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, str(model_dir / "resume.pth"))
    train_time = (time.time() - start_time) / 60
    logger.info(f'finish training, total training time: {train_time:.2f} mins')


if __name__ == "__main__":
    main()



