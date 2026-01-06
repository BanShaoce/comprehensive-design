import tensorflow as tf
from tensorflow.keras.datasets import mnist  # 改用Keras内置数据集
from network import Net
import matplotlib.pyplot as plt
import numpy as np

print('Loading data......')
# 使用Keras加载MNIST数据集，返回训练集和测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 由于Keras默认不返回验证集，这里从训练集中拆分出验证集（调整为2000+200）
train_images = x_train[:15000]  # 训练集调整为15000个样本
validation_images = x_train[15000:15200]  # 验证集调整为200个样本
test_images = x_test[:200]  # 测试集调整为200个样本

# 原代码使用了one-hot编码，这里需要对标签进行one-hot转换
train_labels = tf.keras.utils.to_categorical(y_train[:2000], num_classes=10)  # 对应训练集标签
validation_labels = tf.keras.utils.to_categorical(y_train[2000:], num_classes=10)  # 对应验证集标签
test_labels = tf.keras.utils.to_categorical(y_test[:200], num_classes=10)  # 对应测试集标签

# 数据归一化（将像素值从0-255转换为0-1）
train_images = train_images / 255.0
validation_images = validation_images / 255.0
test_images = test_images / 255.0

print('Preparing data......')
# 保持和原代码一致的数据形状转换（reshape为[样本数, 1, 28, 28]）
training_data = train_images.reshape(20000, 1, 28, 28)  # 训练集形状匹配2000样本
training_labels = train_labels
testing_data = test_images.reshape(200, 1, 28, 28)  # 测试集形状匹配200样本
testing_labels = test_labels

print(training_data.shape, training_labels.shape)
print(testing_data.shape, test_labels.shape)
LeNet = Net()

print('Training Lenet......')
LeNet.train(training_data=training_data, training_label=training_labels,
            val_data=validation_images.reshape(validation_images.shape[0], 1, 28, 28),
            val_label=validation_labels,
            batch_size=64, epoch=3, weights_file="pretrained_weights.pkl")

print('Testing Lenet......')
# 优先使用最优模型进行测试
try:
    print('Testing with best pretrained weights......')
    LeNet.test_with_pretrained_weights(testing_data, testing_labels, 200, 'pretrained_weights_best.pkl')
except FileNotFoundError:
    print('Best model not found, using final model......')
    LeNet.test_with_pretrained_weights(testing_data, testing_labels, 200, 'pretrained_weights.pkl')