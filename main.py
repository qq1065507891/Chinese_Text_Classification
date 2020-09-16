import time
import argparse
import torch
import numpy as np
from importlib import import_module
import utils
import train

parse = argparse.ArgumentParser(description='Chinese-Text-Classification')
parse.add_argument('--model', type=str, default='bert', help='bert, bert_cnn, bert_rnn, bert_rcnn,'
                                                             'bert_dpcnn, ERNIE_DPCNN,'
                                                             'ERNIE')

args = parse.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'
    model_name = args.model
    print('model_name: ', model_name)
    # 动态化加载模型
    x = import_module('models.' + model_name)
    # 加载模型的参数
    config = x.Config(dataset)
    # 保持每次运行取得的数据集是一样的
    np.random.seed(5)
    torch.manual_seed(5)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True

    # 加载数据
    start_time = time.time()
    print('加载数据')
    train_data, test_data, dev_data = utils.bulid_dataset(config)
    train_iter = utils.build_iteration(train_data, config)
    test_iter = utils.build_iteration(test_data, config)
    dev_iter = utils.build_iteration(dev_data, config)

    time_idf = utils.get_time_idf(start_time)
    print('数据加载完成, 用时: ', time_idf)

    # 模型的训练, 验证, 测试
    model = x.Model(config).to(config.device)
    train.train(config, model, train_iter, dev_iter)
    train.test(config, model, test_iter)
