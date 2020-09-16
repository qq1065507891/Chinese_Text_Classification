import torch
import time
import torch.nn.functional as F
from sklearn import metrics
import numpy as np
from pytorch_pretrained.optimization import BertAdam
import utils


def train(config, model, train_iter, dev_iter):
    """
    模型的训练
    :param config:
    :param model:
    :param train_iter:
    :param dev_iter:
    :return:
    """
    start_time = time.time()
    # 启动模型的训练模式
    model.train()
    # 拿到所有参数
    param_optimizer = list(model.named_parameters())
    # 定义不需要衰减的参数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight': 0.0}
    ]
    optimizer = BertAdam(params=optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.epochs)

    total_batch = 0  # 记录进行多少batch
    dev_best_loss = float('inf')  # 记录上次最好的验证集loss
    last_improve = 0  # 记录上次提升的batch
    flag = False  # 停止位的标志, 是否很久没提升

    for epoch in range(config.epochs):
        print('Epoch [{}/{}]'.format(epoch+1, config.epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()

            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            if total_batch % 50 == 0:  # 每训练50次输出在训练集和验证集上的效果
                torch.cuda.empty_cache()
                true = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                score = metrics.accuracy_score(true, predict)

                dev_acc, dev_loss = evaluate(config, model, dev_iter)

                if dev_best_loss > dev_loss:
                    dev_best_loss = dev_loss

                    torch.save(model.state_dict(), config.model_save)
                    improve = '+'
                    last_improve = total_batch
                else:
                    improve = ''
                time_idf = utils.get_time_idf(start_time)
                msg = 'Iter:{0:>6}, Train Loss:{1:>5.2}, Train ACC:{2:>6.2%}, ' \
                      'Val Loss:{3:>5.2}, Val ACC:{4:>6.2%}, Time:{5}  {6}'
                print(msg.format(total_batch, loss.item(), score, dev_loss, dev_acc, time_idf, improve))
                model.train()
            total_batch = total_batch + 1

            if total_batch - last_improve > config.require_improvement:
                # 在验证集上loss超过1000batch没有下降, 结束训练
                print('在验证集上loss超过1000batch没有下降, 结束训练')
                flag = True
                break
        if flag:
            break


def evaluate(config, model, dev_iter, test=False):
    """
    模型评估
    :param config:
    :param model:
    :param dev_iter:
    :param test:
    :return: acc, loss
    """
    model.eval()
    loss_total = 0
    predicts_all = np.array([], dtype=int)
    labels_all = np.array([], dtype='int')

    with torch.no_grad():
        for dev, labels in dev_iter:
            outputs = model(dev)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            true = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            predicts_all = np.append(predicts_all, predict)
            labels_all = np.append(labels_all, true)

    acc = metrics.accuracy_score(labels_all, predicts_all)

    if test:
        report = metrics.classification_report(labels_all, predicts_all, target_names=config.class_name, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predicts_all)
        return acc, loss_total / len(dev_iter), report, confusion

    return acc, loss_total / len(dev_iter)


def test(config, model, test_iter):
    """
    模型测试
    :param config:
    :param model:
    :param test_iter:
    :return:
    """

    model.load_state_dict(torch.load(config.model_save))
    start_time = time.time()

    acc, loss, report, confusion = evaluate(config, model, test_iter, test=True)

    msg = 'Test Loss:{0:>5.2},Test Acc:{1:>6.2%}'

    print(msg.format(loss, acc))
    print('Precision, Recall and F1-Score')
    print(report)
    print('Confusion Matrix')
    print(confusion)
    time_dif = utils.get_time_idf(start_time)
    print('使用时间:', time_dif)


