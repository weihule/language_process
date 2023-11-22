import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    "TextRNN"
]


class TextRNN(nn.Module):
    def __init__(self, numclasses):
        super(TextRNN, self).__init__()
        # 进行词嵌入,5000个中文单词，每个单词64维，是特征数量
        self.embedding = nn.Embedding(5000, 64)

        # 双向RNN，input_size为每个字的向量维度大小，hidden_size、num_layers自选，bidirectional=True表示双向
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, bidirectional=True)

        self.f1 = nn.Sequential(nn.Linear(256, 128),
                                nn.Dropout(0.5),
                                nn.ReLU())
        self.f2 = nn.Sequential(nn.Linear(128, numclasses),
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


if __name__ == "__main__":
    # inputs = torch.tensor([[0, 1, 2], [1, 0, 2]])
    # embedding = nn.Embedding(3, 3)
    # outputs = embedding(inputs)

    model = TextRNN(numclasses=10)
    inputs = torch.randint(low=0, high=200, size=(32, 600))
    outputs = model(inputs)
    print(outputs.shape)

