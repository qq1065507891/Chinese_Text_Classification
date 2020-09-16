import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):
    """
    参数配置类
    """
    def __init__(self, dataset):
        # 模型名称
        self.model_name = 'bert_cnn'
        # 训练数据地址
        self.train_path = dataset + '/data/train.txt'
        # 测试数据地址
        self.test_path = dataset + '/data/test.txt'
        # 验证数据地址
        self.dev_path = dataset + '/data/dev.txt'
        # 类别名
        with open(dataset + '/data/class.txt', 'r') as f:
            self.class_name = [line.strip() for line in f.readlines()]
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
        self.learning_rate = 1e-4
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
        # kernel_size
        self.kernel_size = [1, 2, 3]
        # 卷积核
        self.num_filters = 256


class Model(nn.Module):
    """
    Bert + CNN
    """
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=config.num_filters, kernel_size=(k, config.bert_hidden))
            for k in config.kernel_size
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.kernel_size), config.num_classes)

    def _conv_and_pool(self, x, conv):
        x = conv(x)
        x = F.relu(x)
        x = x.squeeze(3)
        size = x.size(2)
        x = F.max_pool1d(x, size)
        x = x.squeeze(2)
        return x

    def forward(self, x):
        content = x[0]
        mask = x[2]
        encoder_out, text_cls = self.bert(content, attention_mask=mask, output_all_encoded_layers=False)
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self._conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
