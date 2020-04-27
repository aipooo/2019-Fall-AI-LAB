import numpy as np
import csv
import matplotlib.pyplot as plt


def PLA(train_set, cycle, alpha):
    '''
    PLA函数输入为训练集，迭代次数cycle，学习率alpha
    使用PLA算法计算出最终的增广权向量W
    '''
    #初始化W为零向量
    W = np.zeros((1, len(train_set[0])))
    X = train_set[:, :-1]
    X = np.insert(X, 0, np.ones(len(X)), axis=1)
    Y = train_set[:, -1:]
    #迭代cycle次
    for cycle_index in range(cycle):
        #变量is_finished用于判断是否满足迭代停止的条件
        is_finished = True
        for i in range(len(X)):
            if np.sign(Y[i]) != np.sign(np.dot(W, X[i])):
                W = W + alpha * Y[i] * X[i]
                is_finished = False
                break
        if is_finished:
            break
    return W


def classify(W, test_set):
    '''
    利用增广权向量W对测试集进行分类
    函数返回测试集的分类预测结果列表
    '''
    predict_Y = []
    test_set = np.insert(test_set, 0, np.ones(len(test_set)), axis=1)
    for item in test_set:
        label = np.sign(np.dot(W, item))
        predict_Y.append(label)
    return predict_Y


def cal_accuracy(W, test_set, cycle):
    '''
    计算预测准确率
    '''
    predict_Y = classify(W, test_set[:, :-1])
    real_Y = test_set[:, -1:]
    count = 0
    for index in range(len(real_Y)):
        if predict_Y[index] == real_Y[index]:
            count += 1
    return count / len(real_Y)


if __name__ == '__main__':
    file = open(r'C:\Users\Administrator\Desktop\人工智能实验\check_train.csv', 'r')
    file_csv = csv.reader(file)
    file_list = list(file_csv)
    data_set = np.array(file_list)
    data_set = data_set.astype(np.float)
    for row in data_set:
        if row[-1] == 0:
            row[-1] = -1
    train_set = data_set[:]

    tfile = open(r'C:\Users\Administrator\Desktop\人工智能实验\check_test.csv', 'r')
    tfile_csv = csv.reader(tfile)
    tfile_list = list(tfile_csv)
    tdata_set = np.array(tfile_list)
    tdata_set = tdata_set.astype(np.float)
    test_set = tdata_set[:,:-1]


    cycle = 100
    alpha = 1
    W = PLA(train_set, cycle, alpha)
    result = classify(W, test_set)
    for i in range(len(result)):
        print(result[i])
