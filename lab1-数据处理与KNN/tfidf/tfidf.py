import numpy as np

#读取文件，对文本进行处理，从而得到文本词汇表和总单词向量
file = open('C:/Users/Administrator/Desktop/人工智能/lab1 数据处理与KNN/lab1_data/semeval.txt', 'r')
word_matrix = []
word_list = []
line_count = 0
for line in file:
    line_count += 1
    line = line.replace('\n', '')
    line = line.replace('\t', ' ')
    line = line.split(' ')
    word_matrix.append(line[8:])
    for item in line[8:]:
        if item not in word_list:
            word_list.append(item)

#遍历词汇表，统计总单词向量中每个单词在每一个文本中的个数，从而得到One-Hot矩阵
one_hot_matrix = np.zeros((line_count, len(word_list)))
for row in range(len(word_matrix)):
    for col in range(len(word_list)):
        one_hot_matrix[row][col] = word_matrix[row].count(word_list[col])
        
#将One-Hot矩阵中每一个训练文本所对应的向量中单词出现的个数作归一化处理，得到TF矩阵
tf_matrix = one_hot_matrix.copy()
for row in range(len(tf_matrix)):
    tf_matrix[row] /= sum(one_hot_matrix[row])
    
#根据One-Hot矩阵利用公式计算得到IDF矩阵
idf_matrix = one_hot_matrix.sum(axis=0)
idf_matrix = np.log10(len(one_hot_matrix) / idf_matrix)

#将TF矩阵的每一行和IDF矩阵进行点乘，从而得到TF-IDF矩阵
tfidf_matrix = tf_matrix * idf_matrix

#文件输出
np.savetxt("17341178_xueweihao_TFIDF.txt", tfidf_matrix, fmt='%.6f')
file.close()