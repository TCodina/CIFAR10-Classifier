import numpy as np
import matplotlib.pyplot as plt
import torch


def show_data(images, labels, mean=None, std=None, ncols=None):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # If images are tensor unormalize them and reorder axis
    if type(images) == torch.Tensor:
        # Bring to form compatible with images
        mean_t = torch.Tensor(mean).reshape(-1, 1, 1)
        std_t = torch.Tensor(std).reshape(-1, 1, 1)
        # Unormalize and bring channel to last position to plot
        images = (images*std_t + mean_t).permute(0, 2, 3, 1)
    
    # Catch case of single image
    if images.ndim == 3:
        
        plt.subplots(figsize=(10, 2))    
        plt.imshow(images)
        plt.title(classes[labels[0]])
        plt.axis('off')
        plt.show()
        return
    
    elif images.shape[0] == 1:
        
        plt.subplots(figsize=(10, 2))    
        plt.imshow(images[0])
        plt.title(classes[labels[0, 0]])
        plt.axis('off')
        plt.show()
        return
    
    n_imgs = images.shape[0]
                    
    if not ncols:
        factors = [i for i in range(1, n_imgs + 1) if n_imgs % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else n_imgs // 4 + 1
    nrows = int(n_imgs / ncols) + int(n_imgs % ncols)
    imgs = [images[i] if n_imgs > i else None for i in range(nrows * ncols)]

    fig, ax = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows))
    ax = ax.flatten()[:len(imgs)]
    for i in range(n_imgs):
        ax[i].imshow(imgs[i])
        ax[i].set_title(classes[labels[i, 0]])
        ax[i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_metrics(metrics_train, metrics_val, w = 50):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

    ax1.plot(metrics_train['loss'], 'r-', label='train_loss', alpha=0.2)
    ax1.plot(metrics_train['loss'].rolling(w).mean(), 'r-')
    ax1.plot(metrics_val['loss'], 'b-o', label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_xticks(metrics_val.index)
    ax1.legend()
    ax1.grid()

    ax2.plot(metrics_train['acc'], 'r-', label='train_accuracy', alpha=0.2)
    ax2.plot(metrics_train['acc'].rolling(w).mean(), 'r-')
    ax2.plot(metrics_val['acc'], 'b-o', label='val_accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_xticks(metrics_val.index)
    ax2.legend()
    ax2.grid()

    plt.show()
