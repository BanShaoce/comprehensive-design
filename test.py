"""
模型测试模块 - 用于评估 LeNet CNN 模型的性能

使用方法：
    from test import ModelTester
    
    # 加载预训练模型
    tester = ModelTester(weights_file='pretrained_weights_best.pkl')
    
    # 计算准确率
    accuracy = tester.test_accuracy(test_data, test_labels)
    
    # 详细指标
    metrics = tester.test_metrics(test_data, test_labels)
    
    # 每个类别的准确率
    tester.per_class_accuracy(test_data, test_labels)
    
    # 可视化预测
    tester.visualize_predictions(test_data, test_labels, num_samples=10)
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
from network import Net


class ModelTester:
    """模型测试类 - 用于评估 LeNet 模型"""
    
    def __init__(self, weights_file=None):
        """
        初始化测试器
        
        Args:
            weights_file: 预训练权重文件路径
        """
        self.net = Net()
        self.weights_file = weights_file
        if weights_file:
            self.load_weights(weights_file)
    
    def load_weights(self, weights_file):
        """加载预训练权重"""
        try:
            with open(weights_file, 'rb') as handle:
                b = pickle.load(handle)
            self.net.layers[0].feed(b[0]['conv1.weights'], b[0]['conv1.bias'])
            self.net.layers[3].feed(b[3]['conv3.weights'], b[3]['conv3.bias'])
            self.net.layers[6].feed(b[6]['conv5.weights'], b[6]['conv5.bias'])
            self.net.layers[9].feed(b[9]['fc6.weights'], b[9]['fc6.bias'])
            self.net.layers[11].feed(b[11]['fc7.weights'], b[11]['fc7.bias'])
            print(f'✓ Successfully loaded weights from {weights_file}')
        except FileNotFoundError:
            print(f'✗ Weights file {weights_file} not found!')
            raise
    
    def predict_single(self, image):
        """
        预测单个样本
        
        Args:
            image: 输入图像 (1, 28, 28)
            
        Returns:
            digit: 预测数字 (0-9)
            probability: 预测概率
        """
        x = image
        for l in range(self.net.lay_num):
            output = self.net.layers[l].forward(x)
            x = output
        digit = np.argmax(output)
        probability = output[0, digit]
        return digit, probability
    
    def predict_batch(self, images):
        """
        批量预测
        
        Args:
            images: 输入图像数组 (N, 1, 28, 28)
            
        Returns:
            predictions: 预测结果数组 (N,)
            probabilities: 预测概率数组 (N,)
        """
        predictions = []
        probabilities = []
        for img in images:
            digit, prob = self.predict_single(img)
            predictions.append(digit)
            probabilities.append(prob)
        return np.array(predictions), np.array(probabilities)
    
    def test_accuracy(self, test_data, test_label, verbose=True):
        """
        计算模型准确率
        
        Args:
            test_data: 测试数据 (N, 1, 28, 28)
            test_label: 测试标签 (N, 10) one-hot编码
            verbose: 是否打印结果
            
        Returns:
            accuracy: 模型准确率
        """
        test_size = test_data.shape[0]
        predictions, _ = self.predict_batch(test_data)
        true_labels = np.argmax(test_label, axis=1)
        
        correct = np.sum(predictions == true_labels)
        accuracy = correct / test_size
        
        if verbose:
            print(f'=== Test Accuracy: {accuracy:.4f} ({correct}/{test_size}) ===')
        
        return accuracy
    
    def test_metrics(self, test_data, test_label):
        """
        计算详细的评估指标 (精准率、召回率、F1)
        
        Args:
            test_data: 测试数据
            test_label: 测试标签
            
        Returns:
            metrics: 包含各项指标的字典
        """
        predictions, probabilities = self.predict_batch(test_data)
        true_labels = np.argmax(test_label, axis=1)
        
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': true_labels
        }
        
        print('\n' + '='*50)
        print('          MODEL EVALUATION METRICS')
        print('='*50)
        print(f'Accuracy:  {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall:    {recall:.4f}')
        print(f'F1 Score:  {f1:.4f}')
        print('='*50 + '\n')
        
        return metrics
    
    def confusion_matrix_report(self, test_data, test_label, save_path=None):
        """
        生成混淆矩阵可视化
        
        Args:
            test_data: 测试数据
            test_label: 测试标签
            save_path: 保存图像的路径（可选）
        """
        predictions, _ = self.predict_batch(test_data)
        true_labels = np.argmax(test_label, axis=1)
        
        cm = confusion_matrix(true_labels, predictions, labels=range(10))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'✓ Confusion matrix saved to {save_path}')
        
        plt.show()
        return cm
    
    def per_class_accuracy(self, test_data, test_label):
        """
        计算每个数字类别的准确率
        
        Args:
            test_data: 测试数据
            test_label: 测试标签
        """
        predictions, _ = self.predict_batch(test_data)
        true_labels = np.argmax(test_label, axis=1)
        
        print('\n' + '='*40)
        print('   PER-CLASS ACCURACY REPORT')
        print('='*40)
        
        for digit in range(10):
            mask = true_labels == digit
            if np.sum(mask) > 0:
                class_acc = np.sum(predictions[mask] == digit) / np.sum(mask)
                class_count = np.sum(mask)
                print(f'Digit {digit}: {class_acc:.4f} ({np.sum(predictions[mask] == digit)}/{class_count})')
        
        print('='*40 + '\n')
    
    def visualize_predictions(self, test_data, test_label, num_samples=10, save_path=None):
        """
        可视化预测结果
        
        Args:
            test_data: 测试数据
            test_label: 测试标签
            num_samples: 显示样本数
            save_path: 保存图像的路径（可选）
        """
        predictions, probabilities = self.predict_batch(test_data[:num_samples])
        true_labels = np.argmax(test_label[:num_samples], axis=1)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i in range(min(num_samples, 10)):
            ax = axes[i]
            image = test_data[i, 0, :, :]
            
            ax.imshow(image, cmap='gray')
            
            # 预测正确则标题为绿色，错误为红色
            title_text = f'True: {true_labels[i]}, Pred: {predictions[i]}\nProb: {probabilities[i]:.2f}'
            if predictions[i] == true_labels[i]:
                ax.set_title(title_text, color='green', fontweight='bold')
            else:
                ax.set_title(title_text, color='red', fontweight='bold')
            
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'✓ Visualization saved to {save_path}')
        
        plt.show()
    
    def compare_models(self, model_files, test_data, test_label):
        """
        对比多个模型的性能
        
        Args:
            model_files: 模型文件路径列表
            test_data: 测试数据
            test_label: 测试标签
            
        Returns:
            results: 对比结果列表
        """
        results = []
        
        print('\n' + '='*60)
        print('           MODEL COMPARISON REPORT')
        print('='*60)
        
        for model_file in model_files:
            try:
                tester = ModelTester(weights_file=model_file)
                accuracy = tester.test_accuracy(test_data, test_label, verbose=False)
                results.append({
                    'model': model_file,
                    'accuracy': accuracy
                })
                print(f'{model_file:<40} Accuracy: {accuracy:.4f}')
            except FileNotFoundError:
                print(f'{model_file:<40} NOT FOUND')
        
        print('='*60 + '\n')
        
        # 找出最优模型
        if results:
            best_model = max(results, key=lambda x: x['accuracy'])
            print(f'✓ Best model: {best_model["model"]} (Accuracy: {best_model["accuracy"]:.4f})\n')
        
        return results


if __name__ == '__main__':
    print('ModelTester module loaded successfully')
    print('Usage: from test import ModelTester')
