# 模型测试模块 (test.py)

## 概述
`test.py` 是一个独立的模型测试模块，用于评估 LeNet CNN 模型在 MNIST 数据集上的性能。

## 主要功能

### 1. **ModelTester 类**

#### 初始化
```python
from test import ModelTester

# 加载预训练模型
tester = ModelTester(weights_file='pretrained_weights_best.pkl')
```

#### 主要方法

##### `test_accuracy(test_data, test_label, verbose=True)`
计算模型在测试集上的准确率。

**参数：**
- `test_data`: 测试数据，形状为 (N, 1, 28, 28)
- `test_label`: 测试标签，形状为 (N, 10) one-hot编码
- `verbose`: 是否打印结果，默认为 True

**返回：**
- 准确率 (float)

**示例：**
```python
accuracy = tester.test_accuracy(test_data, test_labels)
```

##### `test_metrics(test_data, test_label)`
计算详细的评估指标（精准率、召回率、F1分数）。

**返回：**
- 包含以下键的字典：
  - `accuracy`: 准确率
  - `precision`: 精准率
  - `recall`: 召回率
  - `f1`: F1分数
  - `predictions`: 预测结果数组
  - `probabilities`: 预测概率数组
  - `true_labels`: 真实标签数组

**示例：**
```python
metrics = tester.test_metrics(test_data, test_labels)
print(f"精准率: {metrics['precision']:.4f}")
print(f"F1分数: {metrics['f1']:.4f}")
```

##### `per_class_accuracy(test_data, test_label)`
计算并打印每个数字类别（0-9）的准确率。

**示例：**
```python
tester.per_class_accuracy(test_data, test_labels)
# 输出：
# Digit 0: 0.9500 (19/20)
# Digit 1: 0.9000 (18/20)
# ...
```

##### `predict_single(image)`
预测单个样本。

**参数：**
- `image`: 单个图像，形状为 (1, 28, 28)

**返回：**
- `digit`: 预测数字 (0-9)
- `probability`: 预测概率

**示例：**
```python
digit, prob = tester.predict_single(test_data[0])
print(f"预测: {digit}, 概率: {prob:.4f}")
```

##### `predict_batch(images)`
批量预测。

**参数：**
- `images`: 图像数组，形状为 (N, 1, 28, 28)

**返回：**
- `predictions`: 预测结果数组，形状为 (N,)
- `probabilities`: 预测概率数组，形状为 (N,)

**示例：**
```python
predictions, probs = tester.predict_batch(test_data)
```

##### `confusion_matrix_report(test_data, test_label, save_path=None)`
生成并显示混淆矩阵可视化。

**参数：**
- `test_data`: 测试数据
- `test_label`: 测试标签
- `save_path`: 可选，保存图像的路径

**示例：**
```python
tester.confusion_matrix_report(test_data, test_labels, save_path='confusion_matrix.png')
```

##### `visualize_predictions(test_data, test_label, num_samples=10, save_path=None)`
可视化预测结果，显示预测正确（绿色）和错误（红色）的样本。

**参数：**
- `test_data`: 测试数据
- `test_label`: 测试标签
- `num_samples`: 要显示的样本数，默认为 10
- `save_path`: 可选，保存图像的路径

**示例：**
```python
tester.visualize_predictions(test_data, test_labels, num_samples=10, 
                             save_path='predictions.png')
```

##### `compare_models(model_files, test_data, test_label)`
对比多个模型的性能。

**参数：**
- `model_files`: 模型文件路径列表
- `test_data`: 测试数据
- `test_label`: 测试标签

**返回：**
- 对比结果列表

**示例：**
```python
models = ['pretrained_weights_best.pkl', 'pretrained_weights.pkl']
results = tester.compare_models(models, test_data, test_labels)
```

## 使用示例

### 基本使用

```python
from test import ModelTester
import numpy as np

# 假设已加载测试数据
test_data, test_labels = load_test_data()

# 创建测试器
tester = ModelTester(weights_file='pretrained_weights_best.pkl')

# 计算准确率
accuracy = tester.test_accuracy(test_data, test_labels)

# 获取详细指标
metrics = tester.test_metrics(test_data, test_labels)

# 显示每个类别的准确率
tester.per_class_accuracy(test_data, test_labels)

# 可视化预测
tester.visualize_predictions(test_data, test_labels, num_samples=10)

# 生成混淆矩阵
tester.confusion_matrix_report(test_data, test_labels)
```

### 完整示例（见 test_example.py）

```python
python test_example.py
```

## 测试模块的优势

✅ **独立运行** - 不依赖主训练脚本  
✅ **灵活性** - 可以测试不同的模型  
✅ **详细指标** - 提供多种评估指标  
✅ **可视化** - 支持可视化预测结果和混淆矩阵  
✅ **模型对比** - 可以对比多个模型  
✅ **轻量级** - 只依赖基本的 Python 库  

## 支持的模型文件

- `pretrained_weights_best.pkl` - 最优模型（基于验证集准确率）
- `pretrained_weights.pkl` - 最终模型（训练完成后的模型）

## 输出文件

测试时可以生成以下可视化文件：
- `predictions_best.png` - 最优模型的预测可视化
- `confusion_matrix_best.png` - 最优模型的混淆矩阵
- `predictions_final.png` - 最终模型的预测可视化
- `confusion_matrix_final.png` - 最终模型的混淆矩阵

## 依赖

- numpy
- scikit-learn
- seaborn
- matplotlib

## 注意事项

1. 测试数据必须是 numpy 数组，形状为 (N, 1, 28, 28)
2. 标签必须是 one-hot 编码，形状为 (N, 10)
3. 像素值应该在 0-1 之间（归一化）
4. 模型文件必须通过 `network.train()` 生成
