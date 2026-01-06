"""
evaluate_accuracy.py

下载 MNIST 测试集（如果本地不存在），加载项目里的预训练权重文件并计算准确率。

用法（PowerShell）:
D:/anaconda/envs/CNN/python.exe evaluate_accuracy.py --num_samples 200

"""
import os
import argparse
import gzip
import urllib.request
import numpy as np
from test import ModelTester

MNIST_URLS = {
    't10k-images-idx3-ubyte.gz': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
}


def download_mnist(target_dir='data'):
    os.makedirs(target_dir, exist_ok=True)
    for fname, url in MNIST_URLS.items():
        path = os.path.join(target_dir, fname)
        if not os.path.exists(path):
            print(f'Downloading {fname}...')
            urllib.request.urlretrieve(url, path)
            print('done')
        else:
            print(f'{fname} already exists')
    return target_dir


def load_mnist_test_from_raw(target_dir='data', num_samples=200):
    # First try sklearn.fetch_openml (more reliable), then fall back to raw download
    try:
        from sklearn.datasets import fetch_openml
        print('Trying to load MNIST via sklearn.fetch_openml...')
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int64)
        X = X.reshape(-1, 28, 28)
        if num_samples > X.shape[0]:
            num_samples = X.shape[0]
        images = X[:num_samples]
        labels = y[:num_samples]
        images = images.reshape(num_samples, 1, 28, 28)
        labels_onehot = np.eye(10)[labels]
        return images, labels_onehot
    except Exception as e:
        print('sklearn.fetch_openml failed or unavailable:', e)
        # Ensure raw files exist and try raw download
        download_mnist(target_dir)
        images_path = os.path.join(target_dir, 't10k-images-idx3-ubyte.gz')
        labels_path = os.path.join(target_dir, 't10k-labels-idx1-ubyte.gz')

        with gzip.open(labels_path, 'rb') as lbpath:
            lbpath.read(8)  # header
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

        with gzip.open(images_path, 'rb') as imgpath:
            imgpath.read(16)  # header
            images = np.frombuffer(imgpath.read(), dtype=np.uint8)
            images = images.reshape(-1, 28, 28)

        if num_samples > labels.shape[0]:
            num_samples = labels.shape[0]
        images = images[:num_samples]
        labels = labels[:num_samples]

        images = images.astype(np.float32) / 255.0
        images = images.reshape(num_samples, 1, 28, 28)
        labels_onehot = np.eye(10)[labels]

        return images, labels_onehot


def choose_weights_file():
    if os.path.exists('pretrained_weights_best.pkl'):
        return 'pretrained_weights_best.pkl'
    elif os.path.exists('pretrained_weights.pkl'):
        return 'pretrained_weights.pkl'
    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=200, help='Number of test samples to evaluate')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to download MNIST raw files')
    args = parser.parse_args()

    print('Preparing MNIST test data...')
    try:
        images, labels = load_mnist_test_from_raw(args.data_dir, args.num_samples)
    except Exception as e:
        print('Failed to download/load raw MNIST:', e)
        raise

    weights_file = choose_weights_file()
    if weights_file is None:
        print('No pretrained weights found (pretrained_weights_best.pkl or pretrained_weights.pkl). Please run training first.')
        return

    print(f'Using weights file: {weights_file}')
    tester = ModelTester(weights_file=weights_file)
    acc = tester.test_accuracy(images, labels)
    print(f'Final reported accuracy on {images.shape[0]} samples: {acc:.4f}')


if __name__ == '__main__':
    main()
