import numpy as np
import csv

train_file = open('C:/Users/Administrator/Desktop/人工智能/lab1 数据处理与KNN/lab1_data/regression_dataset/train_set.csv', 'r')
train_csv = csv.reader(train_file)
train_list = list(train_csv)
train_list = train_list[1:]

validation_file = open(
    'C:/Users/Administrator/Desktop/人工智能/lab1 数据处理与KNN/lab1_data/regression_dataset/validation_set.csv', 'r')
validation_csv = csv.reader(validation_file)
validation_list = list(validation_csv)
validation_list = validation_list[1:]

word_list = []
word_matrix = []
for row in train_list:
    row_word_list = row[0].split()
    word_matrix.append(row_word_list)
    for item in row_word_list:
        if item not in word_list:
            word_list.append(item)

validation_word_matrix = []
for row in validation_list:
    row_word_list = row[0].split()
    validation_word_matrix.append(row_word_list)
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

val_one_hot_matrix = np.zeros((len(validation_word_matrix), len(word_list)))
for row in range(len(validation_word_matrix)):
    for col in range(len(word_list)):
        val_one_hot_matrix[row][col] = validation_word_matrix[row].count(word_list[col])

train_tf_matrix = train_one_hot_matrix.copy()
for row in range(len(train_tf_matrix)):
    train_tf_matrix[row] /= sum(train_one_hot_matrix[row])

val_tf_matrix = val_one_hot_matrix.copy()
for row in range(len(val_tf_matrix)):
    val_tf_matrix[row] /= sum(val_one_hot_matrix[row])

train_idf_matrix = train_one_hot_matrix.sum(axis=0)
train_idf_matrix = np.log10(len(train_one_hot_matrix) / (1 + train_idf_matrix))

val_idf_matrix = val_one_hot_matrix.sum(axis=0)
val_idf_matrix = np.log10(len(val_one_hot_matrix) / (1 + val_idf_matrix))

train_tfidf_matrix = train_tf_matrix * train_idf_matrix
val_tfidf_matrix = val_tf_matrix * val_idf_matrix

# (i,j)表示测试集i到训练集j的距离
p = 1
dist_matrix = np.zeros((len(validation_word_matrix), len(word_matrix)))
for val_item, i in zip(val_tfidf_matrix, range(len(val_tfidf_matrix))):
    for train_item, j in zip(train_tfidf_matrix, range(len(train_tfidf_matrix))):
        #闵可夫斯基距离
        #dist_matrix[i][j]=pow((sum(abs(pow(val_item-train_item,p)))),1/p)
        dist_matrix[i][j]=np.linalg.norm(val_item-train_item,ord=3)
        #余弦距离
        #dist_matrix[i][j] = 1 - np.dot(val_item, train_item) / (np.linalg.norm(val_item) * (np.linalg.norm(train_item)))

reality = np.zeros((len(validation_list), 6))
for row_index in range(len(reality)):
    for col_index in range(1, len(validation_list[0])):
        reality[row_index][col_index - 1] = float(validation_list[row_index][col_index])
reality = np.transpose(reality)


for K in range(1, 101):
    dist_index_matrix = np.argsort(dist_matrix, axis=1)
    k_dist_index_matrix = dist_index_matrix[:, :K]
    predicts = np.zeros((len(k_dist_index_matrix), 6))
    for mood_index in range(1, 7):
        k_probability_matrix = np.zeros(np.shape(k_dist_index_matrix))
        for i in range(len(k_dist_index_matrix)):
            for j in range(K):
                if dist_matrix[i][k_dist_index_matrix[i][j]] == 0:
                    k_probability_matrix[i][j] = float(train_list[k_dist_index_matrix[i][j]][mood_index]) / 1e-6
                else:
                    k_probability_matrix[i][j] = float(train_list[k_dist_index_matrix[i][j]][mood_index]) / \
                                                 dist_matrix[i][k_dist_index_matrix[i][j]]
        for row_index in range(len(k_probability_matrix)):
            predicts[row_index][mood_index - 1] = k_probability_matrix[row_index].sum()
    # 归一化
    for row_index in range(len(predicts)):
        row_sum = predicts[row_index].sum()
        for col_index in range(6):
            predicts[row_index][col_index] /= row_sum
    predicts = np.transpose(predicts)

    r_sum = 0
    for row_index in range(len(predicts)):
        tmp = np.corrcoef(predicts[row_index], reality[row_index])
        r_sum += tmp[0][1]

    r_average = r_sum / len(predicts)
    print(K, ':', r_average)

