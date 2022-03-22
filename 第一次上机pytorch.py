import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import torchmetrics
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision      # 数据库模块
import matplotlib.pyplot as plt
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 50
LR = 0.001          # 学习率
train_dataset = torchvision.datasets.MNIST(root='./mnist',
                           train=True,
                           transform=transforms.ToTensor(),
                           download=False)

test_dataset = torchvision.datasets.MNIST(root='./mnist',
                           train=False,
                           transform=transforms.ToTensor())

#参数设置
input_size = 784       # The image size = 28 x 28 = 784
hidden_size = 500      # The number of nodes at the hidden layer
num_classes = 10       # The number of output classes. In this case, from 0 to 9
num_epochs = 5         # The number of times entire dataset is trained
batch_size = 100       # The size of input data took for one iteration
learning_rate = 0.001  # The speed of convergence

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
#前馈神经网络模型
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        '''
        :param input_size:
        :param hidden_size:
        :param num_classes:
        '''
        super(Net, self).__init__()                    # Inherited from the parent class nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer: 784 (input data) -> 500 (hidden node)
        self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = nn.Linear(hidden_size, num_classes) # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)

    def forward(self, x):                              # Forward pass: stacking each layer together
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
#---------------------------------------------------
#定义一个分类数*分类数的混淆矩阵
# Emotion_kinds=10
# conf_matrix = torch.zeros(Emotion_kinds, Emotion_kinds)
# def confusion_matrix(preds, labels, conf_matrix):
#     preds = torch.argmax(preds, 1)
#     for p, t in zip(preds, labels):
#         conf_matrix[p, t] += 1
#     return conf_matrix


# net1 = torch.nn.Sequential(
#     torch.nn.Linear(784, 500),
#     torch.nn.ReLU(),
#     torch.nn.Linear(500, 10)
# )

# 将保存的参数复制到 net3
# net1.load_state_dict(torch.load('fnn_model.pkl'))
#混淆矩阵显示精确度
# net1=torch.load("fnn.pkl")
# with torch.no_grad():
#     for step, (imgs, targets) in enumerate(test_loader):
#         # imgs:     torch.Size([50, 3, 200, 200])   torch.FloatTensor
#         # targets:  torch.Size([50, 1]),     torch.LongTensor  多了一维，所以我们要把其去掉
#         targets = targets.squeeze()  # [50,1] ----->  [50]
#         imgs = Variable(imgs.view(-1, 28 * 28))
#
#         # # 将变量转为gpu
#         # targets = targets.cuda()
#         # imgs = imgs.cuda()
#         # print(step,imgs.shape,imgs.type(),targets.shape,targets.type())
#
#         out = net1(imgs)
#         # 记录混淆矩阵参数
#         acc = torchmetrics.functional.accuracy(out,targets)
#         print(acc)
#         conf_matrix = confusion_matrix(out, targets, conf_matrix)
#         conf_matrix = conf_matrix.cpu()
# # conf_matrix=np.array(conf_matrix.cpu())# 将混淆矩阵从gpu转到cpu再转到np
# corrects=conf_matrix.diagonal(offset=0)#抽取对角线的每种分类的识别正确个数
# per_kinds=conf_matrix.sum(axis=1)#抽取每个分类数据总的测试条数
# print(conf_matrix)
# # print("混淆矩阵总元素个数：{0}".format(int(np.sum(conf_matrix))))
# # 获取每种Emotion的识别准确率
# print("每种数字总个数：",per_kinds)
# print("每种数字预测正确的个数：",corrects)
# print("每种数字的识别准确率为：{0}".format([rate*100 for rate in corrects/per_kinds]))
def test_loop(dataloader, model, loss_fn):
    # 实例化相关metrics的计算对象
    test_acc = torchmetrics.Accuracy()
    test_recall = torchmetrics.Recall(average='none', num_classes=10)
    test_precision = torchmetrics.Precision(average='none', num_classes=10)
    test_auc = torchmetrics.AUROC(average="macro", num_classes=10)

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X=Variable(X.view(-1, 28 * 28))
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # 一个batch进行计算迭代
            test_acc(pred.argmax(1), y)
            test_auc.update(pred, y)
            test_recall(pred.argmax(1), y)
            test_precision(pred.argmax(1), y)

    test_loss /= num_batches
    correct /= size

    # 计算一个epoch的accuray、recall、precision、AUC
    total_acc = test_acc.compute()
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_auc = test_auc.compute()
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, "
          f"Avg loss: {test_loss:>8f}, "
          f"torch metrics acc: {(100 * total_acc):>0.1f}%\n")
    print("recall of every test dataset class: ", total_recall)
    print("precision of every test dataset class: ", total_precision)
    print("auc:", total_auc.item())

    # 清空计算对象
    test_precision.reset()
    test_acc.reset()
    test_recall.reset()
    test_auc.reset()

net1=torch.load("fnn.pkl")
loss=nn.CrossEntropyLoss()
test_loop(test_loader,net1,loss)
#----------------------------------------------------
# net = Net(input_size, hidden_size, num_classes)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):   # Load a batch of images with its (index, data, class)
#         images = Variable(images.view(-1, 28*28))         # Convert torch tensor to Variable: change image from a vector of size 784 to a matrix of 28 x 28
#         labels = Variable(labels)
#
#         optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
#         outputs = net(images)                             # Forward pass: compute the output class given a image
#         loss = criterion(outputs, labels)                 # Compute the loss: difference between the output class and the pre-given label
#         loss.backward()                                   # Backward pass: compute the weight
#         optimizer.step()                                  # Optimizer: update the weights of hidden nodes
#
#         if (i+1) % 100 == 0:                              # Logging
#             print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
#                  %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))
#
# correct = 0
# total = 0
# for images, labels in test_loader:
#     images = Variable(images.view(-1, 28*28))
#     outputs = net(images)
#     _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
#     total += labels.size(0)                    # Increment the total count
#     correct += (predicted == labels).sum()     # Increment the correct count
#
# print('Accuracy of the network on the 10K test images: %d %%' % (100 * correct / total))
# torch.save(net, 'fnn.pkl')

#定义一个softmax模型，只有输入层和输出层
class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        # 定义一个输入层

    # 定义向前传播（在这个两层网络中，它也是输出层）
    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        # 将x换形为y后，再继续向前传播
        return y

# net=LinearNet(784,10)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):   # Load a batch of images with its (index, data, class)
#         images = Variable(images.view(-1, 28*28))         # Convert torch tensor to Variable: change image from a vector of size 784 to a matrix of 28 x 28
#         labels = Variable(labels)
#
#         optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
#         outputs = net(images)                             # Forward pass: compute the output class given a image
#         loss = criterion(outputs, labels)                 # Compute the loss: difference between the output class and the pre-given label
#         loss.backward()                                   # Backward pass: compute the weight
#         optimizer.step()                                  # Optimizer: update the weights of hidden nodes
#
#         if (i+1) % 100 == 0:                              # Logging
#             print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
#                  %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))
#
# correct = 0
# total = 0
# for images, labels in test_loader:
#     images = Variable(images.view(-1, 28*28))
#     outputs = net(images)
#     _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
#     total += labels.size(0)                    # Increment the total count
#     correct += (predicted == labels).sum()     # Increment the correct count
#
# print('Accuracy of the network on the 10K test images: %d %%' % (100 * correct / total))
# torch.save(net, 'fnn_softmax.pkl')
# #计算准确率
# def net_accurary(data_iter,net):
#     right_sum,n = 0.0,0
#     for X,y in data_iter:
#     #从迭代器data_iter中获取X和y
#         right_sum += (net(X).argmax(dim=1)==y).float().sum().item()
#         #计算准确判断的数量
#         n +=y.shape[0]
#         #通过shape[0]获取y的零维度（列）的元素数量
#     return right_sum/n

