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
        self.model_name = 'ERNIE_DPCNN'
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
        self.bert_path = 'ERNIE_pretrain'
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
        # rnn 隐藏层
        self.num_filters = 256


class Model(nn.Module):
    """
    Bert + DPCNN
    """
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.bert_hidden))

        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1))

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)

        self.padd1 = nn.ZeroPad2d([0, 0, 1, 1])
        self.padd2 = nn.ZeroPad2d([0, 0, 0, 1])

        self.relu = nn.ReLU()

        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def _block(self, x):
        x = self.padd2(x)
        px = self.max_pool(x)

        x = self.padd1(px)
        x = self.relu(x)
        x = self.conv(x)

        x = self.padd1(x)
        x = self.relu(x)
        x = self.conv(x)

        x = x + px
        return x

    def forward(self, x):
        content = x[0]
        mask = x[2]
        encoder_out, text_cls = self.bert(content, attention_mask=mask, output_all_encoded_layers=False)
        out = encoder_out.unsqueeze(1)

        out = self.conv_region(out)
        out = self.padd1(out)
        out = self.relu(out)

        out = self.conv(out)
        out = self.padd1(out)
        out = self.relu(out)

        out = self.conv(out)

        while out.size()[2] > 2:
            out = self._block(out)

        out = out.squeeze()
        out = self.fc(out)
        return out
