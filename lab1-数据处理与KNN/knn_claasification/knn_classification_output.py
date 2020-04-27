import numpy as np
import pandas as pd
import csv

train_file = open('C:/Users/Administrator/Desktop/人工智能/lab1 数据处理与KNN/lab1_data/classification_dataset/train_set.csv', 'r')
train_csv = csv.reader(train_file)
train_list = list(train_csv)
train_list = train_list[1:]

validation_file = open('C:/Users/Administrator/Desktop/人工智能/lab1 数据处理与KNN/lab1_data/classification_dataset/validation_set.csv', 'r')
validation_csv = csv.reader(validation_file)
validation_list = list(validation_csv)
validation_list = validation_list[1:]
for row in validation_list:
    train_list.append(row)

test_file = open('C:/Users/Administrator/Desktop/人工智能/lab1 数据处理与KNN/lab1_data/classification_dataset/test_set.csv', 'r')
test_csv = csv.reader(test_file)
test_list = list(test_csv)
test_list = test_list[1:]
test_list = [row[1:] for row in test_list]

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
    row_word_list = row[0].split()
    test_word_matrix.append(row_word_list)
    for item in row_word_list:
        if item not in word_list:
            word_list.append(item)

label_dict = {'joy': 1,
              'sad': 2,
              'fear': 3,
              'anger': 4,
              'surprise': 5,
              'disgust': 6
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
dist_matrix = np.zeros((len(test_tfidf_matrix), len(train_tfidf_matrix)))
for test_item, i in zip(test_tfidf_matrix, range(len(test_tfidf_matrix))):
    for train_item, j in zip(train_tfidf_matrix, range(len(train_tfidf_matrix))):
        #闵可夫斯基距离
        #dist_matrix[i][j]=pow((sum(abs(pow(val_item-train_item,p)))),1/p)
        #dist_matrix[i][j]=np.linalg.norm(val_item-train_item,ord=2)
        #余弦距离
        dist_matrix[i][j] = 1 - np.dot(test_item, train_item) / (np.linalg.norm(test_item) * (np.linalg.norm(train_item)))

K = 10
#对距离矩阵的每一行按照从小到大的顺序排序，从而得到一个序号矩阵
#该矩阵的坐标（i,j）表示验证集向量i在训练集向量空间中距离第j小的训练集向量的序号
dist_index_matrix = np.argsort(dist_matrix, axis=1)
#选取序号矩阵的前K列，得到新的序号矩阵
#该矩阵记录了验证集空间的每一个向量到训练集空间距离最小的K个向量的所对应的序号
k_dist_index_matrix = dist_index_matrix[:, :K]

#获取序号矩阵中每一个序号所对应的情绪标签，得到一个标签矩阵
k_label_matrix = np.zeros(np.shape(k_dist_index_matrix))
for i in range(len(k_dist_index_matrix)):
    for j in range(K):
        k_label_matrix[i][j] = label_dict[train_list[k_dist_index_matrix[i][j]][1]]
k_label_matrix = k_label_matrix.astype(np.int32)

#统计标签矩阵每一行的众数，该众数即为该验证集向量我们所预测的标签
predicts = np.zeros((len(k_label_matrix), 1))
for row_index in range(len(k_label_matrix)):
    count_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for item in k_label_matrix[row_index]:
        count_dict[item] += 1
    predicts[row_index] = max(count_dict, key=count_dict.get)
predicts = predicts.astype(np.int32)

reverse_dict = {v: k for (k, v) in label_dict.items()}
predicts = [reverse_dict[predicts[i][0]] for i in range(len(predicts))]
test_list = [item[0] for item in test_list]
predicts = np.transpose(np.array(predicts))
test_list = np.transpose(np.array(test_list))
final_predicts = pd.DataFrame(columns=['Words (split by space)'], data=test_list, index=range(1, len(test_list)+1))
final_predicts['label'] = predicts
# print(final_predicts)
final_predicts.to_csv('17341178_xuewiehao_KNN_classification.csv')
