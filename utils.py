from tqdm import tqdm
import os
import pickle as pkl
import torch
import time
from datetime import timedelta


PAD, CLS = '[PAD]', '[CLS]'


def load_dataset(file_path, tokenizer, pad_size):
    """
    加载数据集
    :param file_path:
    :param config:
    :return:[list ids, label, ids_len, mask]
    """
    contents = []
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            content, label = line.split('\t')
            token = tokenizer.tokenize(content)
            token = [CLS] + token
            seq_len = len(token)
            mask = []

            token_ids = tokenizer.convert_tokens_to_ids(token)
            if pad_size:
                if seq_len < pad_size:
                    mask = [1] * len(token) + [0] * (pad_size - len(token))
                    token_ids = token_ids + [0] * (pad_size - len(token))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, int(label), seq_len, mask))
    return contents


def bulid_dataset(config):
    """
    建立dataset, 返回
    :param config:
    :return: train, test, dev
    """
    if os.path.exists(config.datasetpkl):
        with open(config.datasetpkl, 'rb') as f:
            dataset = pkl.load(f)
        train = dataset['train']
        test = dataset['test']
        dev = dataset['dev']
    else:
        train = load_dataset(config.train_path, config.tokenizer, config.pad_size)
        test = load_dataset(config.test_path, config.tokenizer, config.pad_size)
        dev = load_dataset(config.dev_path, config.tokenizer, config.pad_size)
        dataset = {}
        dataset['train'] = train
        dataset['test'] = test
        dataset['dev'] = dev
        with open(config.datasetpkl, 'wb') as f:
            pkl.dump(dataset, f)
    return train, test, dev


class DatasetIterator(object):
    """
    数据迭代器
    """
    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.residue = False  # 记录n_batch数是否为整数
        self.n_batches = len(dataset) // batch_size
        self.index = 0
        if len(dataset) % self.n_batches != 0:
            self.residue = True

    def _to_tensor(self, datas):
        x = torch.LongTensor([item[0] for item in datas]).to(self.device)
        y = torch.LongTensor([item[1] for item in datas]).to(self.device)

        seq_len = torch.LongTensor([item[2] for item in datas]).to(self.device)
        mask = torch.LongTensor([item[3] for item in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size: len(self.dataset)]
            self.index = self.index + 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index = self.index + 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iteration(dataset, config):
    """生成数据迭代器"""
    iter = DatasetIterator(dataset, config.batch_size, config.device)
    return iter


def get_time_idf(start_time):
    """
    获取已经使用的时间
    :param start_time:
    :return: 返回使用多长时间
    """
    end_time = time.time()
    time_idf = end_time - start_time
    return timedelta(seconds=int(round(time_idf)))

# https://tf.wiki/zh_hans/
# https://github.com/datawhalechina
