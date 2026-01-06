"""
compare_with_pytorch.py

Usage examples (PowerShell):

# Basic run (requires PyTorch + torchvision installed and a saved .pt model)
python compare_with_pytorch.py --pytorch_model path\\to\\pytorch_model.pt --our_model pretrained_weights_best.pkl --num_samples 200

# If you can't import torch, you can precompute PyTorch predictions (numpy) and then compare using --pytorch_preds path.npy
"""

import argparse
import numpy as np
import pickle
import os
from test import ModelTester
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_mnist_numpy(num_samples=200, root=None):
    """Load MNIST test images as numpy arrays (N,1,28,28) and one-hot labels.
    This function tries torchvision first; if not available it falls back to keras if present.
    """
    try:
        import torchvision
        from torchvision import transforms
        from torchvision.datasets import MNIST
        import torch
        transform = transforms.Compose([transforms.ToTensor()])
        # download to local folder if needed
        ds = MNIST(root=root or './data', train=False, download=True, transform=transform)
        imgs = []
        labels = []
        for i in range(min(len(ds), num_samples)):
            img, label = ds[i]
            # img is tensor (1,28,28)
            imgs.append(img.numpy())
            labels.append(label)
        images = np.stack(imgs, axis=0).astype(np.float32)
        images = images  # already [0,1]
        labels_onehot = np.eye(10)[labels]
        return images, labels_onehot
    except Exception:
        # fallback to tensorflow.keras
        try:
            import tensorflow as tf
            from tensorflow.keras.datasets import mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            test_images = x_test[:num_samples]
            test_labels = tf.keras.utils.to_categorical(y_test[:num_samples], num_classes=10)
            test_images = test_images / 255.0
            testing_data = test_images.reshape(num_samples, 1, 28, 28).astype(np.float32)
            return testing_data, test_labels
        except Exception as e:
            raise RuntimeError('Neither torchvision nor tensorflow is available to load MNIST.\n' + str(e))


def evaluate_and_report(name, preds, true_labels, save_prefix=None):
    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds, average='weighted', zero_division=0)
    rec = recall_score(true_labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, preds, average='weighted', zero_division=0)
    print(f"== {name} ==")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}\n")

    if save_prefix:
        cm = confusion_matrix(true_labels, preds, labels=range(10))
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        out = save_prefix + f'_{name}_confusion.png'
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f'Confusion matrix saved to {out}')

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def pytorch_predict_from_model(model_path, images, device='cpu', input_key='input'):
    """Load a PyTorch model (.pt/.pth) and run inference on images (numpy array N,1,28,28).
    The function expects the model to accept an input tensor shape (N,1,28,28) and output logits or probabilities.
    If the saved object is a state_dict, user should provide a small wrapper model.
    """
    try:
        import torch
    except Exception as e:
        raise RuntimeError('PyTorch not available: ' + str(e))

    model_obj = torch.load(model_path, map_location=device)
    # If saved entire model
    if isinstance(model_obj, torch.nn.Module):
        model = model_obj
    else:
        # could be state_dict or dict
        # try to handle common cases: if it's a dict with 'model_state_dict' or similar, user should adjust
        raise RuntimeError('Loaded object is not a torch.nn.Module. Please save your full model (torch.save(model)).')

    model.to(device)
    model.eval()
    X = torch.from_numpy(images).to(device)
    with torch.no_grad():
        outputs = model(X)
        # outputs shape (N, num_classes)
        if outputs.ndim == 4:
            # maybe returns images; flatten last dims
            outputs = outputs.view(outputs.shape[0], -1)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pytorch_model', type=str, help='Path to PyTorch model (.pt)')
    parser.add_argument('--pytorch_preds', type=str, help='Path to precomputed numpy predictions (.npy) or (preds.npy,probs.npy)')
    parser.add_argument('--our_model', type=str, default='pretrained_weights_best.pkl', help='Our model weights file (.pkl)')
    parser.add_argument('--num_samples', type=int, default=200, help='Number of MNIST test samples to use')
    parser.add_argument('--save_prefix', type=str, default='compare', help='Prefix for saved plots')
    parser.add_argument('--device', type=str, default='cpu', help='Device for PyTorch (cpu or cuda)')
    args = parser.parse_args()

    print('Loading test data...')
    images, labels_onehot = load_mnist_numpy(num_samples=args.num_samples)
    true_labels = np.argmax(labels_onehot, axis=1)
    print(f'Loaded {images.shape[0]} samples')

    # Our model predictions
    print('\nRunning inference with our model...')
    our_tester = ModelTester(weights_file=args.our_model)
    our_preds, our_probs = our_tester.predict_batch(images)
    our_metrics = evaluate_and_report('OurModel', our_preds, true_labels, save_prefix=args.save_prefix)

    # PyTorch predictions
    pytorch_preds = None
    pytorch_probs = None
    if args.pytorch_preds:
        # allow either single .npy with preds or a .npz with preds and probs
        if args.pytorch_preds.endswith('.npz'):
            d = np.load(args.pytorch_preds)
            pytorch_preds = d['preds']
            pytorch_probs = d['probs'] if 'probs' in d else None
        else:
            pytorch_preds = np.load(args.pytorch_preds)
        print(f'Loaded PyTorch predictions from {args.pytorch_preds}')
    elif args.pytorch_model:
        print('Running PyTorch model inference...')
        try:
            pytorch_preds, pytorch_probs = pytorch_predict_from_model(args.pytorch_model, images, device=args.device)
        except Exception as e:
            print('PyTorch inference failed:', e)
            print('You can alternatively precompute predictions in PyTorch and provide via --pytorch_preds')
            return
    else:
        print('No PyTorch model or predictions provided, skipping PyTorch side.')

    if pytorch_preds is not None:
        pytorch_metrics = evaluate_and_report('PyTorchModel', pytorch_preds, true_labels, save_prefix=args.save_prefix)

        # Compare directly
        print('\nDirect comparison:')
        print(f"OurModel Accuracy: {our_metrics['accuracy']:.4f} | PyTorch Accuracy: {pytorch_metrics['accuracy']:.4f}")

        # Save side-by-side confusion matrix figure
        cm_our = confusion_matrix(true_labels, our_preds, labels=range(10))
        cm_pt = confusion_matrix(true_labels, pytorch_preds, labels=range(10))
        fig, axes = plt.subplots(1,2, figsize=(14,6))
        sns.heatmap(cm_our, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('OurModel')
        sns.heatmap(cm_pt, annot=True, fmt='d', cmap='Blues', ax=axes[1])
        axes[1].set_title('PyTorchModel')
        out = args.save_prefix + '_side_by_side_confusion.png'
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f'Side-by-side confusion saved to {out}')

    print('\nDone.')


if __name__ == '__main__':
    main()
