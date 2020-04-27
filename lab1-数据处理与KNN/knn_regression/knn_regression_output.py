import numpy as np
import csv
import pandas as pd

train_file = open('C:/Users/Administrator/Desktop/人工智能/lab1 数据处理与KNN/lab1_data/regression_dataset/train_set.csv', 'r')
train_csv = csv.reader(train_file)
train_list = list(train_csv)
train_list = train_list[1:]

validation_file = open('C:/Users/Administrator/Desktop/人工智能/lab1 数据处理与KNN/lab1_data/regression_dataset/validation_set.csv', 'r')
validation_csv = csv.reader(validation_file)
validation_list = list(validation_csv)
validation_list = validation_list[1:]
for row in validation_list:
    train_list.append(row)

test_file = open('C:/Users/Administrator/Desktop/人工智能/lab1 数据处理与KNN/lab1_data/regression_dataset/test_set.csv', 'r')
test_csv = csv.reader(test_file)
test_list = list(test_csv)
test_list = test_list[1:]

word_list = []
word_matrix = []
for row in train_list:
    row_word_list = row[0].split()
    word_matrix.append(row_word_list)
    for item in row_word_list:
        if item not in word_list:
            word_list.append(item)

test_word_matrix = []
for row in test_list:
    row_word_list = row[1].split()
    test_word_matrix.append(row_word_list)
    for item in row_word_list:
        if item not in word_list:
            word_list.append(item)

label_dict = {'anger': 1,
              'disgust': 2,
              'fear': 3,
              'joy': 4,
              'sad': 5,
              'surprise': 6
              }

train_one_hot_matrix = np.zeros((len(word_matrix), len(word_list)))
for row in range(len(word_matrix)):
    for col in range(len(word_list)):
        train_one_hot_matrix[row][col] = word_matrix[row].count(word_list[col])

test_one_hot_matrix = np.zeros((len(test_word_matrix), len(word_list)))
for row in range(len(test_word_matrix)):
    for col in range(len(word_list)):
        test_one_hot_matrix[row][col] = test_word_matrix[row].count(word_list[col])

train_tf_matrix = train_one_hot_matrix.copy()
for row in range(len(train_tf_matrix)):
    train_tf_matrix[row] /= sum(train_one_hot_matrix[row])

test_tf_matrix = test_one_hot_matrix.copy()
for row in range(len(test_tf_matrix)):
    test_tf_matrix[row] /= sum(test_one_hot_matrix[row])

train_idf_matrix = train_one_hot_matrix.sum(axis=0)
train_idf_matrix = np.log10(len(train_one_hot_matrix) / (1 + train_idf_matrix))

test_idf_matrix = test_one_hot_matrix.sum(axis=0)
test_idf_matrix = np.log10(len(test_one_hot_matrix) / (1 + test_idf_matrix))

train_tfidf_matrix = train_tf_matrix * train_idf_matrix
test_tfidf_matrix = test_tf_matrix * test_idf_matrix

# (i,j)表示测试集i到训练集j的距离
p = 1
dist_matrix = np.zeros((len(test_word_matrix), len(word_matrix)))
for val_item, i in zip(test_tfidf_matrix, range(len(test_tfidf_matrix))):
    for train_item, j in zip(train_tfidf_matrix, range(len(train_tfidf_matrix))):
        #闵可夫斯基距离
        #dist_matrix[i][j]=pow((sum(abs(pow(val_item-train_item,p)))),1/p)
        #dist_matrix[i][j]=np.linalg.norm(val_item-train_item,ord=1)
        #余弦距离
        dist_matrix[i][j] = 1 - np.dot(val_item, train_item) / (np.linalg.norm(val_item) * (np.linalg.norm(train_item)))

#获取训练集各种情绪的实际概率矩阵
reality = np.zeros((len(validation_list), 6))
for row_index in range(len(reality)):
    for col_index in range(1, len(validation_list[0])):
        reality[row_index][col_index - 1] = float(validation_list[row_index][col_index])
reality = np.transpose(reality)

K=5
#对距离矩阵的每一行按照从小到大的顺序排序，从而得到一个序号矩阵
#该矩阵的坐标（i,j）表示验证集向量i在训练集向量空间中距离第j小的训练集向量的序号
dist_index_matrix = np.argsort(dist_matrix, axis=1)
#选取序号矩阵的前K列，得到新的序号矩阵
#该矩阵记录了验证集空间的每一个向量到训练集空间距离最小的K个向量的所对应的序号
k_dist_index_matrix = dist_index_matrix[:, :K]
#获取序号矩阵中每一个序号所对应的情绪的概率，得到该情绪所对应的概率矩阵
#由于有六种情绪，所以可以得到六个这样的矩阵
predicts = np.zeros((len(k_dist_index_matrix), 6))
#根据公式进行计算
for mood_index in range(1, 7):
    k_probability_matrix = np.zeros(np.shape(k_dist_index_matrix))
    for i in range(len(k_dist_index_matrix)):
        for j in range(K):
            #如果两个向量的距离为0，则将该距离视为1e-6
            if dist_matrix[i][k_dist_index_matrix[i][j]] == 0:
                k_probability_matrix[i][j] = float(train_list[k_dist_index_matrix[i][j]][mood_index]) / 1e-6
            else:
                k_probability_matrix[i][j] = float(train_list[k_dist_index_matrix[i][j]][mood_index]) / dist_matrix[i][k_dist_index_matrix[i][j]]
    #得到加权计算后的该情绪所对应的概率（未归一化）
    for row_index in range(len(k_probability_matrix)):
        predicts[row_index][mood_index - 1] = k_probability_matrix[row_index].sum()

#对每一个文本向量的六种情绪的预测概率进行归一化处理
for row_index in range(len(predicts)):
    row_sum = predicts[row_index].sum()
    for col_index in range(6):
        predicts[row_index][col_index] /= row_sum

test_list=[item[1] for item in test_list]
test_list=np.array(test_list)
test_list=test_list.transpose()
output = pd.DataFrame(columns=['Words (split by space)'], data=test_list,index=range(1,len(test_list)+1))
output['anger']=predicts[:,0]
output['disgust']=predicts[:,1]
output['fear']=predicts[:,2]
output['joy']=predicts[:,3]
output['sad']=predicts[:,4]
output['surprise']=predicts[:,5]
output.to_csv('17341178_xuewiehao_KNN_regression.csv', float_format='%.6f')
