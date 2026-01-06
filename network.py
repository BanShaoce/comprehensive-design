import numpy as np
import pickle
import sys
from time import *
from conv_layer import ConvolutionLayer
from fully_connected_layer import FullyConnectedLayer
from maxpooling_layer import MaxPoolingLayer
from softmax import Softmax
from relu import ReLu
from loss import cross_entropy
from flatten import Flatten


class Net:
    def __init__(self):
        # Lenet网络结构（保持不变）
        lr = 0.01
        self.layers = []
        self.layers.append(
            ConvolutionLayer(inputs_channel=1, num_filters=6, width=5, height=5, padding=2, stride=1, learning_rate=lr,
                             name='conv1'))
        self.layers.append(ReLu())
        self.layers.append(MaxPoolingLayer(width=2, height=2, stride=2, name='maxpool2'))
        self.layers.append(
            ConvolutionLayer(inputs_channel=6, num_filters=16, width=5, height=5, padding=0, stride=1, learning_rate=lr,
                             name='conv3'))
        self.layers.append(ReLu())
        self.layers.append(MaxPoolingLayer(width=2, height=2, stride=2, name='maxpool4'))
        self.layers.append(
            ConvolutionLayer(inputs_channel=16, num_filters=120, width=5, height=5, padding=0, stride=1,
                             learning_rate=lr,
                             name='conv5'))
        self.layers.append(ReLu())
        self.layers.append(Flatten())
        self.layers.append(FullyConnectedLayer(num_inputs=120, num_outputs=84, learning_rate=lr, name='fc6'))
        self.layers.append(ReLu())
        self.layers.append(FullyConnectedLayer
                           (num_inputs=84, num_outputs=10, learning_rate=lr, name='fc7'))
        self.layers.append(Softmax())
        self.lay_num = len(self.layers)

    def train(self, training_data, training_label, batch_size, epoch, weights_file, val_data=None, val_label=None):
        total_acc = 0
        total_samples = training_data.shape[0]  # 总样本数（如5500）
        best_acc = 0
        best_weights_file = weights_file.replace('.pkl', '_best.pkl')
        
        for e in range(epoch):
            # 按批次迭代，步长为batch_size
            for batch_index in range(0, total_samples, batch_size):
                # 计算当前批次的实际样本数（最后一批可能不足batch_size）
                end_idx = min(batch_index + batch_size, total_samples)
                current_batch_size = end_idx - batch_index  # 关键：获取实际批次大小
                data = training_data[batch_index:end_idx]
                label = training_label[batch_index:end_idx]

                loss = 0
                acc = 0
                start_time = time()

                # 循环当前批次的实际样本数（而非固定的batch_size）
                for b in range(current_batch_size):
                    x = data[b]
                    y = label[b]
                    # 前向传播
                    for l in range(self.lay_num):
                        output = self.layers[l].forward(x)
                        x = output
                    loss += cross_entropy(output, y)
                    # 计算准确率
                    if np.argmax(output) == np.argmax(y):
                        acc += 1
                        total_acc += 1
                    # 反向传播
                    dy = y
                    for l in range(self.lay_num - 1, -1, -1):
                        dout = self.layers[l].backward(dy)
                        dy = dout

                # 时间计算
                end_time = time()
                batch_time = end_time - start_time
                # 剩余迭代次数 = 总迭代数 - 已完成迭代数
                completed = e * total_samples + end_idx
                remaining = epoch * total_samples - completed
                remain_time = (remaining / current_batch_size) * batch_time  # 按实际批次大小估算时间
                hrs = int(remain_time // 3600)
                mins = int((remain_time % 3600) // 60)
                secs = int(remain_time % 60)

                # 结果计算（修正准确率分母）
                loss /= current_batch_size
                batch_acc = float(acc) / current_batch_size
                training_acc = float(total_acc) / completed  # 已完成样本数作为分母

                print(
                    '=== Epoch: {0:d}/{1:d} === Iter:{2:d} === Loss: {3:.2f} === BAcc: {4:.2f} === TAcc: {5:.2f} === Remain: {6:d} Hrs {7:d} Mins {8:d} Secs ==='.format(
                        e, epoch, end_idx, loss, batch_acc, training_acc, hrs, mins, secs
                    )
                )
            
            # 每个epoch后进行验证
            if val_data is not None and val_label is not None:
                val_acc = self.validate(val_data, val_label)
                print('=== Epoch: {0:d}/{1:d} === Validation Acc: {2:.2f} ==='.format(e, epoch, val_acc))
                
                # 如果验证准确率更好，保存最优模型
                if val_acc > best_acc:
                    best_acc = val_acc
                    obj = []
                    for i in range(self.lay_num):
                        cache = self.layers[i].extract()
                        obj.append(cache)
                    with open(best_weights_file, 'wb') as handle:
                        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print('*** Best model saved! Accuracy: {0:.2f} ***'.format(best_acc))

        # 保存最终权重
        obj = []
        for i in range(self.lay_num):
            cache = self.layers[i].extract()
            obj.append(cache)
        with open(weights_file, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('*** Final model saved! ***')

    def validate(self, val_data, val_label):
        """验证模型准确率（不进行反向传播）"""
        val_size = val_data.shape[0]
        total_acc = 0
        
        for i in range(val_size):
            x = val_data[i]
            y = val_label[i]
            
            # 前向传播
            for l in range(self.lay_num):
                output = self.layers[l].forward(x)
                x = output
            
            # 计算准确率
            if np.argmax(output) == np.argmax(y):
                total_acc += 1
        
        val_acc = float(total_acc) / float(val_size)
        return val_acc

    # 以下test、test_with_pretrained_weights、predict_with_pretrained_weights方法保持不变
    def test(self, data, label, test_size):
        toolbar_width = 40
        sys.stdout.write("[%s]" % (" " * (toolbar_width - 1)))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width))
        step = float(test_size) / float(toolbar_width)
        st = 1
        total_acc = 0
        for i in range(test_size):
            if i == round(step):
                step += float(test_size) / float(toolbar_width)
                st += 1
                sys.stdout.write(".")
                sys.stdout.flush()
            x = data[i]
            y = label[i]
            for l in range(self.lay_num):
                output = self.layers[l].forward(x)
                x = output
            if np.argmax(output) == np.argmax(y):
                total_acc += 1
        sys.stdout.write("\n")
        print('=== Test Size:{0:d} === Test Acc:{1:.2f} ==='.format(test_size, float(total_acc) / float(test_size)))

    def test_with_pretrained_weights(self, data, label, test_size, weights_file):
        with open(weights_file, 'rb') as handle:
            b = pickle.load(handle)
        self.layers[0].feed(b[0]['conv1.weights'], b[0]['conv1.bias'])
        self.layers[3].feed(b[3]['conv3.weights'], b[3]['conv3.bias'])
        self.layers[6].feed(b[6]['conv5.weights'], b[6]['conv5.bias'])
        self.layers[9].feed(b[9]['fc6.weights'], b[9]['fc6.bias'])
        self.layers[11].feed(b[11]['fc7.weights'], b[11]['fc7.bias'])
        toolbar_width = 40
        sys.stdout.write("[%s]" % (" " * (toolbar_width - 1)))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width))
        step = float(test_size) / float(toolbar_width)
        st = 1
        total_acc = 0
        for i in range(test_size):
            if i == round(step):
                step += float(test_size) / float(toolbar_width)
                st += 1
                sys.stdout.write(".")
                sys.stdout.flush()
            x = data[i]
            y = label[i]
            for l in range(self.lay_num):
                output = self.layers[l].forward(x)
                x = output
            if np.argmax(output) == np.argmax(y):
                total_acc += 1
        sys.stdout.write("\n")
        print('=== Test Size:{0:d} === Test Acc:{1:.2f} ==='.format(test_size, float(total_acc) / float(test_size)))

    def predict_with_pretrained_weights(self, inputs, weights_file):
        with open(weights_file, 'rb') as handle:
            b = pickle.load(handle)
        self.layers[0].feed(b[0]['conv1.weights'], b[0]['conv1.bias'])
        self.layers[3].feed(b[3]['conv3.weights'], b[3]['conv3.bias'])
        self.layers[6].feed(b[6]['conv5.weights'], b[6]['conv5.bias'])
        self.layers[9].feed(b[9]['fc6.weights'], b[9]['fc6.bias'])
        self.layers[11].feed(b[11]['fc7.weights'], b[11]['fc7.bias'])
        for l in range(self.lay_num):
            output = self.layers[l].forward(inputs)
            inputs = output
        digit = np.argmax(output)
        probability = output[0, digit]
        return digit, probability