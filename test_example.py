"""
模型测试使用示例

这个脚本展示如何使用 ModelTester 进行完整的模型测试
"""

import numpy as np
from test import ModelTester
import tensorflow as tf
from tensorflow.keras.datasets import mnist


def load_mnist_test_data(num_samples=200):
    """加载 MNIST 测试数据"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # 准备测试数据
    test_images = x_test[:num_samples]
    test_labels = tf.keras.utils.to_categorical(y_test[:num_samples], num_classes=10)
    
    # 数据归一化
    test_images = test_images / 255.0
    
    # 转换为网络所需的形状 (N, 1, 28, 28)
    testing_data = test_images.reshape(num_samples, 1, 28, 28)
    testing_labels = test_labels
    
    return testing_data, testing_labels


def main():
    print('='*70)
    print('           LENET CNN 模型测试示例')
    print('='*70 + '\n')
    
    # 加载测试数据
    print('1. Loading test data......')
    test_data, test_labels = load_mnist_test_data(num_samples=200)
    print(f'   ✓ Loaded {test_data.shape[0]} test samples\n')
    
    # 测试最优模型
    print('2. Testing Best Model')
    print('-'*70)
    try:
        tester_best = ModelTester(weights_file='pretrained_weights_best.pkl')
        
        # 计算准确率
        accuracy_best = tester_best.test_accuracy(test_data, test_labels)
        
        # 详细指标
        metrics_best = tester_best.test_metrics(test_data, test_labels)
        
        # 每个类别的准确率
        tester_best.per_class_accuracy(test_data, test_labels)
        
        # 可视化预测
        tester_best.visualize_predictions(test_data, test_labels, num_samples=10,
                                         save_path='predictions_best.png')
        
        # 混淆矩阵
        tester_best.confusion_matrix_report(test_data, test_labels,
                                           save_path='confusion_matrix_best.png')
        
    except FileNotFoundError:
        print('✗ Best model not found. Please run training first.\n')
    
    # 测试最终模型
    print('\n3. Testing Final Model')
    print('-'*70)
    try:
        tester_final = ModelTester(weights_file='pretrained_weights.pkl')
        
        # 计算准确率
        accuracy_final = tester_final.test_accuracy(test_data, test_labels)
        
        # 详细指标
        metrics_final = tester_final.test_metrics(test_data, test_labels)
        
        # 每个类别的准确率
        tester_final.per_class_accuracy(test_data, test_labels)
        
        # 可视化预测
        tester_final.visualize_predictions(test_data, test_labels, num_samples=10,
                                          save_path='predictions_final.png')
        
        # 混淆矩阵
        tester_final.confusion_matrix_report(test_data, test_labels,
                                            save_path='confusion_matrix_final.png')
        
    except FileNotFoundError:
        print('✗ Final model not found. Please run training first.\n')
    
    # 模型对比
    print('\n4. Model Comparison')
    print('-'*70)
    try:
        model_files = ['pretrained_weights_best.pkl', 'pretrained_weights.pkl']
        comparison_results = tester_best.compare_models(model_files, test_data, test_labels)
    except Exception as e:
        print(f'✗ Model comparison failed: {e}\n')
    
    print('='*70)
    print('           TEST COMPLETED')
    print('='*70)


if __name__ == '__main__':
    main()
