import numpy as np
import csv

train_file=open('C:/Users/Administrator/Desktop/人工智能/lab1 数据处理与KNN/lab1_data/classification_dataset/train_set.csv','r')
train_csv=csv.reader(train_file)
train_list=list(train_csv)
train_list=train_list[1:]

validation_file=open('C:/Users/Administrator/Desktop/人工智能/lab1 数据处理与KNN/lab1_data/classification_dataset/validation_set.csv','r')
validation_csv=csv.reader(validation_file)
validation_list=list(validation_csv)
validation_list=validation_list[1:]

word_list=[]
word_matrix=[]
for row in train_list:
    row_word_list=row[0].split()
    word_matrix.append(row_word_list)
    for item in row_word_list:
        if item not in word_list:
            word_list.append(item)

validation_word_matrix=[]
for row in validation_list:
    row_word_list=row[0].split()
    validation_word_matrix.append(row_word_list)
    for item in row_word_list:
        if item not in word_list:
            word_list.append(item)

label_dict={'joy':1,
      'sad':2,
      'fear':3,
      'anger':4,
      'surprise':5,
      'disgust':6
      }

train_one_hot_matrix=np.zeros((len(word_matrix),len(word_list)))
for row in range(len(word_matrix)):
    for col in range(len(word_list)):
        train_one_hot_matrix[row][col]=word_matrix[row].count(word_list[col])

val_one_hot_matrix=np.zeros((len(validation_word_matrix),len(word_list)))
for row in range(len(validation_word_matrix)):
    for col in range(len(word_list)):
        val_one_hot_matrix[row][col]=validation_word_matrix[row].count(word_list[col])

#(i,j)表示测试集i到训练集j的距离
p=1        
dist_matrix=np.zeros((len(validation_word_matrix),len(word_matrix)))
for val_item,i in zip(val_one_hot_matrix,range(len(val_one_hot_matrix))):
    for train_item,j in zip(train_one_hot_matrix,range(len(train_one_hot_matrix))):
#        闵可夫斯基距离
#        dist_matrix[i][j]=pow((sum(abs(pow(val_item-train_item,p)))),1/p)
        dist_matrix[i][j]=np.linalg.norm(val_item-train_item,ord=1)
#        余弦距离
#        dist_matrix[i][j]=1-np.dot(val_item,train_item)/(np.linalg.norm(val_item)*(np.linalg.norm(train_item)))

for K in range(1, 101):
    dist_index_matrix = np.argsort(dist_matrix, axis=1)
    k_dist_index_matrix = dist_index_matrix[:, :K]
    k_label_matrix = np.zeros(np.shape(k_dist_index_matrix))
    for i in range(len(k_dist_index_matrix)):
        for j in range(K):
            k_label_matrix[i][j] = label_dict[train_list[k_dist_index_matrix[i][j]][1]]
    k_label_matrix = k_label_matrix.astype(np.int32)

    predicts = np.zeros((len(k_label_matrix), 1))
    for row_index in range(len(k_label_matrix)):
        count_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        for item in k_label_matrix[row_index]:
            count_dict[item] += 1
        predicts[row_index] = max(count_dict, key=count_dict.get)
    predicts = predicts.astype(np.int32)

    reality = np.zeros(np.shape(predicts))
    for index in range(len(validation_list)):
        reality[index][0] = label_dict[validation_list[index][1]]
    reality.astype(np.int32)

    print(K, ' : ', sum(predicts == reality) / len(reality))


