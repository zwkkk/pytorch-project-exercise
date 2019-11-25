# -*- coding: utf-8 -*
import argparse
import torch
import torch.nn as nn
from flyai.dataset import Dataset
from torch.optim import Adam

from model import Model
from net import resnet18
from path import MODEL_PATH
from transformation import src
# 数据获取辅助类
dataset = Dataset()

# 模型操作辅助类
model = Model(dataset)

# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=100, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=24, type=int, help="batch size")
args = parser.parse_args()

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# device = 'cpu'
device = torch.device(device)

def eval(model, x_test, y_test):
    cnn.eval()
    batch_eval = model.batch_iter(x_test, y_test)
    total_acc = 0.0
    data_len = len(x_test)
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        outputs = cnn(x_batch)
        _, prediction = torch.max(outputs.data, 1)
        correct = (prediction == y_batch).sum().item()
        acc = correct / batch_len
        total_acc += acc * batch_len
    return total_acc / data_len


cnn = resnet18().to(device)


#实验改动位置
#################################################################################
'''
针对optimizer进行实验
'''
optimizer = Adam(cnn.parameters(), lr=3e-4)  # 选用AdamOptimizer


#学习率优化若800轮训练loss未下降，则学习率*0.1

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 800, gamma=0.1, last_epoch=-1)

#################################################################################

loss_fn = nn.CrossEntropyLoss()  # 定义损失函数

# 训练并评估模型

data = Dataset()
model = Model(data)

best_accuracy = 0
for i in range(args.EPOCHS):
    cnn.train()
    x_train, y_train, x_test, y_test = data.next_batch(args.BATCH)  # 读取数据

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_train = x_train.float().to(device)
    y_train = y_train.long().to(device)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    x_test = x_test.float().to(device)
    y_test = y_test.long().to(device)

    outputs = cnn(x_train)
    _, prediction = torch.max(outputs.data, 1)

    optimizer.zero_grad()
    # print(x_train.shape,outputs.shape,y_train.shape)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()   #优化器针对loss进行参数更新
    scheduler.step()    #scheduler为针对学习率的调整策略
    print(loss.detach())
    # 若测试准确率高于当前最高准确率，则保存模型
    train_accuracy = eval(model, x_test, y_test)
    if train_accuracy >= best_accuracy:
        best_accuracy = train_accuracy
        model.save_model(cnn, MODEL_PATH, overwrite=True)
        print("step %d, best accuracy %g" % (i, best_accuracy))

    print(str(i) + "/" + str(args.EPOCHS))

print(best_accuracy)
