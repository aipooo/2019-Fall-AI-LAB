import numpy as np
import csv
import matplotlib.pyplot as plt


def sigmoid(X):
    return 1.0 / (1 + np.exp(-X))


def LR(train_set, cycle, alpha):
    '''
    PLA函数输入为训练集，迭代次数cycle，学习率alpha
    使用PLA算法计算出最终的增广权向量W
    '''
    # 初始化W为零向量
    W = np.zeros((1, len(train_set[0])))
    X = train_set[:, :-1]
    X = np.insert(X, 0, np.ones(len(X)), axis=1)
    Y = train_set[:, -1:]
    # 迭代cycle次
    for i in range(cycle):
        gradient = np.dot((Y - sigmoid(np.dot(W, X.transpose())).transpose()).transpose(), X)
        if (abs(gradient) < 1e-9).all():
            break
        W = W + alpha * gradient
    return W


def classify(W, test_set):
    '''
    利用增广权向量W对测试集进行分类
    函数返回测试集的分类预测结果列表
    '''
    predict_Y = []
    test_set = np.insert(test_set, 0, np.ones(len(test_set)), axis=1)
    for item in test_set:
        p = sigmoid(np.dot(W, item))
        if p >= 0.5:
            predict_Y.append(1)
        else:
            predict_Y.append(0)
    return predict_Y


def cal_accuracy(W, test_set, cycle):
    predict_Y = classify(W, test_set[:, :-1])
    real_Y = test_set[:, -1:]
    count = 0
    for index in range(len(real_Y)):
        if abs(predict_Y[index] - real_Y[index]) < 1e-6:
            count += 1
    return count / len(real_Y)


if __name__ == '__main__':
    file = open(r'C:\Users\Administrator\Desktop\人工智能实验\check_train.csv', 'r')
    file_csv = csv.reader(file)
    file_list = list(file_csv)
    data_set = np.array(file_list)
    data_set = data_set.astype(np.float)
    train_set = data_set[:]
    
    tfile = open(r'C:\Users\Administrator\Desktop\人工智能实验\check_test.csv', 'r')
    tfile_csv = csv.reader(tfile)
    tfile_list = list(tfile_csv)
    tdata_set = np.array(tfile_list)
    tdata_set = tdata_set.astype(np.float)
    test_set = tdata_set[:,:-1]


    cycle = 5000
    alpha = 1
    W = LR(train_set, cycle, alpha)
    result = classify(W, test_set)
    for i in range(len(result)):
        print(result[i])
