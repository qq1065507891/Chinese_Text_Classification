import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):
    """
    参数配置类
    """
    def __init__(self, dataset):
        # 模型名称
        self.model_name = 'bert_rnn'
        # 训练数据地址
        self.train_path = dataset + '/data/train.txt'
        # 测试数据地址
        self.test_path = dataset + '/data/test.txt'
        # 验证数据地址
        self.dev_path = dataset + '/data/dev.txt'
        # 类别名
        with open(dataset + '/data/class.txt', 'r') as f:
            self.class_name = [line.strip() for line in f.readlines()]
        # 类别数量
        self.num_classes = len(self.class_name)
        # 有GPU则启动GPU, 否则启动CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 模型保存地址
        self.model_save = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        # 数据集保存地址
        self.datasetpkl = dataset + '/data/datasetpkl.pkl'
        # bert
        self.bert_path = 'bert_pretrain'
        # bert分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # 学习率
        self.learning_rate = 1e-5
        # bert隐藏层个数
        self.bert_hidden = 768
        # batch_size
        self.batch_size = 128
        # epochs
        self.epochs = 3
        # 每句话处理的长度
        self.pad_size = 32
        # 当模型超过1000次没有提升时, 停止
        self.require_improvement = 1000
        # dropout
        self.dropout = 0.5
        # layers 数量
        self.num_layers = 2
        # rnn 隐藏层
        self.rnn_hidden = 128


class Model(nn.Module):
    """
    Bert + RNN
    """
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.lstm = nn.LSTM(config.bert_hidden, config.rnn_hidden, config.num_layers, bidirectional=True,
                            batch_first=True, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.rnn_hidden*2, config.num_classes)

    def forward(self, x):
        content = x[0]
        mask = x[2]
        encoder_out, text_cls = self.bert(content, attention_mask=mask, output_all_encoded_layers=False)
        out, _ = self.lstm(encoder_out)
        out = self.dropout(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
