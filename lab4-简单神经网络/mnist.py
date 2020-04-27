from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.optimizers import RMSprop
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 先看看数据集大小和部分数据
print(train_images.shape)
print(train_labels)
print(test_images.shape)
print(test_labels)

# 网络架构
network = models.Sequential()
'''
网络包含 2 个 Dense 层，它们是密集连接（也叫全连接）的神经层。第二层（也
是最后一层）是一个 10 路 softmax 层，它将返回一个由 10 个概率值（总和为 1）组成的数组。
每个概率值表示当前数字图像属于 10 个数字类别中某一个的概率。
'''
network.add(layers.Dense(units=512, activation='relu', input_shape=(28*28, )))
# 只有第一层需要指明数据大小,后面的自动根据上一层返回数据大小自动推断
network.add(layers.Dense(units=10, activation='softmax'))

# 编译步骤
network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 简单的数据预处理(改变大小和归一化)
train_images = train_images.reshape((60000, 28*28)).astype('float32')/255
test_images = test_images.reshape((10000, 28*28)).astype('float32')/255

# 准备标签
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 训练网络，用fit函数, epochs表示训练多少个回合， batch_size表示每次训练给多大的数据
network.fit(train_images, train_labels, epochs=10, batch_size=128)

# 来在测试集上测试一下模型的性能吧
test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print("test_loss:", test_loss, "    test_accuracy:", test_accuracy)
