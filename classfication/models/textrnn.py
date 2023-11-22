import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    "TextRNN",
    "TextRNN2"
]


class TextRNN(nn.Module):
    def __init__(self, num_classes, vocab_size):
        super(TextRNN, self).__init__()
        # 进行词嵌入,5000个中文单词，每个单词64维，是特征数量
        self.embedding = nn.Embedding(vocab_size, 64)
        self.num_classes = num_classes
        self.vocab_size = vocab_size

        # 双向RNN，input_size为每个字的向量维度大小，hidden_size、num_layers自选，bidirectional=True表示双向
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, bidirectional=True)

        self.f1 = nn.Sequential(nn.Linear(256, 128),
                                nn.Dropout(0.5),
                                nn.ReLU())
        self.f2 = nn.Sequential(nn.Linear(128, num_classes),
                                nn.Softmax(dim=-1))

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = F.dropout(x, 0.5)

        # 变为二维, 进行全连接
        # 这一步相当于选择了第二个维度最后一个元素
        # 即从[b, l, f]中选择了l最后一个, 变成了 [b, f]
        x = x[:, -1, :]
        x = self.f1(x)
        x = self.f2(x)

        return x


# 循环神经网络 (many-to-one)
class TextRNN2(nn.Module):
    def __init__(self, num_classes, vocab_size):
        super(TextRNN2, self).__init__()
        self.num_classes = num_classes
        self.vocab_size = vocab_size

        embedding_dim = 128
        self.hidden_size = 64
        self.layer_num = 2
        self.bidirectional = True

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim,  # x的特征维度,即embedding_dim
                            self.hidden_size,  # 隐藏层单元数
                            self.layer_num,  # 层数
                            batch_first=True,  # 第一个维度设为 batch, 即:(batch_size, seq_length, embedding_dim)
                            bidirectional=self.bidirectional)  # 是否用双向
        self.fc = nn.Linear(self.hidden_size * 2, num_classes) if self.bidirectional else nn.Linear(self.hidden_size,
                                                                                                    num_classes)

    def forward(self, x):
        # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大长度
        x = self.embedding(x)  # 经过embedding,x的维度为(batch_size, time_step, input_size=embedding_dim)

        # 隐层初始化
        # h0维度为(num_layers*direction_num, batch_size, hidden_size)
        # c0维度为(num_layers*direction_num, batch_size, hidden_size)
        h0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size) if self.bidirectional else torch.zeros(
            self.layer_num, x.size(0), self.hidden_size)
        h0 = h0.to(x.device)

        c0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size) if self.bidirectional else torch.zeros(
            self.layer_num, x.size(0), self.hidden_size)
        c0 = c0.to(x.device)

        # LSTM前向传播，此时out维度为(batch_size, seq_length, hidden_size*direction_num)
        # hn,cn表示最后一个状态?维度与h0和c0一样
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # 我们只需要最后一步的输出,即(batch_size, -1, output_size)
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":
    # inputs = torch.tensor([[0, 1, 2], [1, 0, 2]])
    # embedding = nn.Embedding(3, 3)
    # outputs = embedding(inputs)

    model = TextRNN2(num_classes=10)
    inputs = torch.randint(low=0, high=200, size=(32, 600))
    outputs = model(inputs)
    print(outputs.shape)
