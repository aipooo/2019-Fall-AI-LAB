import io
import math
import numpy as np
from matplotlib import pyplot as plt
import keras
from keras.layers import Input, LSTM, Dense, Embedding, Dropout, BatchNormalization, Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize
from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer
from AttentionLayer import AttentionLayer

# 停用词集合
stop_words = set(stopwords.words('english'))
#lemmatizer = WordNetLemmatizer()

# This is an attempt to solve semantic textual similarity 
# Benchmark semeval data is taken..training and testing is done using this data itself
# trian, dev, and test data contain 5749 1500 1379 sentence pairs respectively 
trainData = "stsbenchmark/sts-train.csv"
devData = "stsbenchmark/sts-dev.csv"
testData = "stsbenchmark/sts-test.csv"
glove = "stsbenchmark/glove.6B.50d.txt"
VECTOR_LEN = 50
THRESHOLD = 20

def get_words_after_stopwords(word_list):
    new_word_list = []
    for word in word_list:
        if word not in stop_words:
            new_word_list.append(word)
    return new_word_list

def preprocess(Data, THRESHOLD) :
    '''
    处理csv数据，得到的结果以list方式存储
    list的每一个元素也为list，包括[score, sentence1, sentence2]
    两个句子长度均超过阈值的元素移除出list
    '''
    DataList = []
    with io.open(Data, encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\t')
            for item in line:
                item = item.strip()
            DataList.append(line[4:7]) 
    #将两个句子长度均超过阈值的句子移除出列表
    newDataList = []
    #对句子进行删除停用词后，再进行分词处理
    for i in range(len(DataList))[::-1] :
#        sent1 = get_words_after_stopwords(word_tokenize(DataList[i][1]))
#        sent2 = get_words_after_stopwords(word_tokenize(DataList[i][2]))
        sent1 = word_tokenize(DataList[i][1])
        sent2 = word_tokenize(DataList[i][2])
        if len(sent1)<=THRESHOLD and len(sent2)<=THRESHOLD:
            #score类型转为浮点型
            output = float(DataList[i][0])
            newDataList.append([output, sent1, sent2])
    return newDataList  


# Create a dictionary with ids for all the words/tokens in the corpus
# Giving a common id for all the words not found in glove
def Word2Id(Data, embeddingDict):
    '''
    Data为句子的列表，embedding为单词的向量表示
    函数根据语料库返回一个字典，key为Data里的单词，将单词映射为一个特定Id
    '''
    idDict = {}
    for sentence in Data:
        for word in sentence:
            #如果单词不在字典中
            if idDict.get(word) is None:
                #如果单词不在embeddingDict中，映射值设为-1
                if embeddingDict.get(word) is None:
                    idDict[word] = -1
                #如果word在embeddingDict中，分情况考虑
                else :
                    #如果字典为空，则将token映射值赋为1
                    if len(idDict) == 0:
                        idDict[word] = 1
                    else:
                        #找到字典中最大的value对应的key
                        maxKey = max(idDict, key=idDict.get)
                        highestId = idDict[maxKey]
                        if highestId >= 1 :
                            idDict[word] = highestId + 1
                        else :
                            idDict[word] = 1
    #由于此前所有不在embeddingDict中的单词的映射值均设为了-1
    #现在更新它们的映射值为最大映射值+1
    maxKey = max(idDict, key=idDict.get)
    highestId = idDict[maxKey]
    for word, value in idDict.items():
        if value == -1:
            idDict[word] = highestId + 1
    return idDict


def convert(Data, idDict):
    '''
    将数据集映射为idDict存储
    '''
    for i, sentence in enumerate(Data):
        for j, word in enumerate(sentence):
            Data[i][j] = idDict[word]
    return Data



def correlation_coefficient_loss(a, b):   
    a_avg = sum(a)/len(a)
    b_avg = sum(b)/len(b)
    cov_ab = sum([(x - a_avg)*(y - b_avg) for x,y in zip(a, b)])
    sq = math.sqrt(sum([(x - a_avg)**2 for x in a])*sum([(x - b_avg)**2 for x in b]))
    corr_factor = cov_ab/sq
    return corr_factor





def getEmbeddingDict():
    '''
    加载glove为一个字典，key为单词，value为该词对应的低维向量表示embedding
    '''
    f = open(glove, 'r', encoding='utf-8')
    embeddingDict = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32') 
        embeddingDict[word] = coefs 
    f.close()  
    return embeddingDict

def getVocabSize(idDict):
    maxKey = max(idDict, key=idDict.get)
    highestValue = idDict[maxKey]
    VOCAB_SIZE = highestValue +1   
    return VOCAB_SIZE


def getEmbedMatrix(idDict, embeddingDict):
    #构建embedding矩阵，行号表示对应的单词，该行为对应的embedding
    VOCAB_SIZE = getVocabSize(idDict)
    
    embedMatrix = np.zeros((VOCAB_SIZE,VECTOR_LEN))
    vector = np.random.rand(VECTOR_LEN)
    for key, value in idDict.items():
        if value != 0 :
            embed = embeddingDict.get(key)
            if embed is None:
                embedMatrix[value] = vector 
            else:
                embedMatrix[value] = embed 
    return embedMatrix


def BuildModel(idtrain_a, idtrain_b, trainOP, iddev_a, iddev_b, devOP, idtest_a, idtest_b, testOP, embedMatrix, VOCAB_SIZE):
    #使用Keras函数API进行模型构建
    input_a = Input(shape=(THRESHOLD,))
    input_b = Input(shape=(THRESHOLD,))
    # This embedding layer will encode the input sequence
    # into a sequence of dense THRESHOLD-dimensional vectors.
    #调用Embedding函数
    x = Embedding(input_dim = VOCAB_SIZE,       #词汇表大小
                  output_dim = VECTOR_LEN,      #词向量维度
                  input_length = THRESHOLD,     #输入序列的长度，句子的最大长度
                  weights = [embedMatrix])      #weight指定了初始化的权重参数
    
    #获取两个句子的embedding
    embed_a = x(input_a)
    embed_b = x(input_b)
    # This layer can take as input a matrix and will return a vector of size 64
    #这一层可以输入一个矩阵，返回一个维度为64的向量
    shared_lstm = LSTM(units = 64, return_sequences = True)
    
    #获取两个句子对应的编码
    encode_a = shared_lstm(embed_a)
    encode_b = shared_lstm(embed_b)
    
    attention = AttentionLayer()
    encode_a = attention(encode_a)
    encode_b = attention(encode_b)
    
    
    # We can then concatenate the two vectors:
    merged_vector = keras.layers.concatenate([encode_a, encode_b], axis=-1)
    merged_vector = Dropout(0.2)(merged_vector)
    merged_vector = BatchNormalization()(merged_vector)
    # And add a logistic regression on top
    dense1 = Dense(70, activation='relu')(merged_vector)
    dense2 = Dense(40, activation='relu')(dense1)
    dense3 = Dense(10, activation='relu')(dense2)
    dense3 = BatchNormalization()(dense3)
    predictions = Dense(1, activation='linear')(dense3)        

    # We define a trainable model linking the sentence inputs to the predictions
    model = Model(inputs = [input_a, input_b], 
                  outputs = predictions)
    model.compile(optimizer = 'RMSprop', 
                  loss = 'mse', 
                  metrics = ['accuracy'])
    history = model.fit([idtrain_a, idtrain_b], trainOP, epochs=30, validation_data=([iddev_a, iddev_b], devOP))
    
    result = model.predict([idtest_a, idtest_b], verbose=0)
    loss, score = model.evaluate([idtest_a, idtest_b], testOP, verbose=1)
    print('mse loss:', loss)
    print('corr:', correlation_coefficient_loss(testOP, result))
    
    return history





if __name__ == '__main__':
    train = preprocess(trainData, THRESHOLD)
    dev = preprocess(devData, THRESHOLD)
    test = preprocess(testData, THRESHOLD)
    trainOP = [item[0] for item in train]
    train_a = [item[1] for item in train]
    train_b = [item[2] for item in train]
    devOP = [item[0] for item in dev]
    dev_a = [item[1] for item in dev]
    dev_b = [item[2] for item in dev]
    testOP = [item[0] for item in test]
    test_a = [item[1] for item in test]
    test_b = [item[2] for item in test]
    
    embeddingDict = getEmbeddingDict()
    sentenceList = train_a + train_b + dev_a + dev_b + test_a + test_b
    idDict = Word2Id(sentenceList, embeddingDict)

    #将各个数据集映射到对应的id
    idtrain_a = convert(train_a, idDict)
    idtrain_b = convert(train_b, idDict)
    iddev_a = convert(dev_a, idDict)
    iddev_b = convert(dev_b, idDict)
    idtest_a = convert(test_a, idDict)
    idtest_b = convert(test_b, idDict)

    #padding，让句子等长
    idtrain_a = pad_sequences(idtrain_a, maxlen=THRESHOLD, padding='pre', truncating= 'post', value=0.0)
    idtrain_b = pad_sequences(idtrain_b, maxlen=THRESHOLD, padding='pre', truncating= 'post', value=0.0)
    iddev_a = pad_sequences(iddev_a, maxlen=THRESHOLD, padding='pre', truncating= 'post', value=0.0)
    iddev_b = pad_sequences(iddev_b, maxlen=THRESHOLD, padding='pre', truncating= 'post', value=0.0)
    idtest_a = pad_sequences(idtest_a, maxlen=THRESHOLD,padding='pre', truncating= 'post', value=0.0)
    idtest_b = pad_sequences(idtest_b, maxlen=THRESHOLD,padding='pre', truncating= 'post', value=0.0)
    
    embedMatrix = getEmbedMatrix(idDict, embeddingDict)
    VOCAB_SIZE = getVocabSize(idDict)
    
    history = BuildModel(idtrain_a, idtrain_b, trainOP, iddev_a, iddev_b, devOP, idtest_a, idtest_b, testOP, embedMatrix, VOCAB_SIZE)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('mse loss')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


