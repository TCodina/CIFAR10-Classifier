import random
import pickle
import numpy as np
import torch
import pandas as pd


def load_data(data_dir):
    """
    Load CIFAR-10 training and test datasets from local directory
    and return them in the form of dictionaries.
    It is assumed the files were downloaded from https://www.cs.toronto.edu/~kriz/cifar.html.
    """

    data_train = {'imgs': [], 'labels': []}
    data_test = {'imgs': [], 'labels': []}
    # Load training data split into 5 batches
    for i in range(1, 6):
        with open(data_dir + f'data_batch_{i}', 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        data_train['imgs'].append(batch[bytes('data', 'utf-8')])
        data_train['labels'].append(batch[bytes('labels', 'utf-8')])
    # Reshape images in a unified numpy 4D matrix with num_channels = 3 at the end (convention)
    data_train['imgs'] = np.concatenate(data_train['imgs']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data_train['labels'] = np.array(data_train['labels']).reshape(-1, 1)
    # Load test data
    with open(data_dir + f'test_batch', 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    data_test['imgs'] = np.array(batch[bytes('data', 'utf-8')]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data_test['labels'] = np.array(batch[bytes('labels', 'utf-8')]).reshape(-1, 1)
    # Load name of labels
    with open(data_dir + f'batches.meta', 'rb') as fo:
        labels_list_bytes = pickle.load(fo, encoding='bytes')[bytes('label_names', 'utf-8')]
    labels_list = list(map(lambda x: str(x)[2:-1], labels_list_bytes))  # Convert to string format

    return data_train, data_test, labels_list


def split_train_val(data, val_size):
    """
    Split data into train and validation sets.
    In order the resulting datasets to be balanced, we pick the first val_size/10 
    examples of each class from the original data. 
    """

    assert val_size % 10 == 0, 'val_size must be divisible by 10'
    class_size = val_size//10
    counting = np.zeros(10)  # To get exactly class_size of each class
    indices = []  # Indices of first examples of each class

    for i, label in enumerate(data['labels']):
      if counting[label] < class_size:
        counting[label] += 1
        indices.append(i)
      else:   
        if np.all(counting == class_size):
          break

    data_val = {'imgs': data['imgs'][indices], 
                'labels': data['labels'][indices]}

    data_train = {'imgs': np.delete(data['imgs'], indices, axis=0), 
                  'labels': np.delete(data['labels'], indices, axis=0)}

    return data_train, data_val


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"Random seed set as {seed}")


def get_accuracy(y_pred, y):
    tp = (y_pred.argmax(axis=1) == y.squeeze()).sum()  # True positives
    return tp.item() / len(y)  # Mean over batches


def test_model(loader, model, device):
    """
    Run model over testset and returns a dataframe with labels and predictions
    """
    test_dict = {'labels': [], 
                 'predictions': [],
                 'correct': []
                 }

    with torch.no_grad():
        loop = tqdm(loader)
        loop.set_description(f"Testing")
        for x, y in loop:
            x = x.to(device)
            y = y.to(device).squeeze()

            # Infer
            y_pred_prob = model(x)
            # Convert one-hot-encoding to list of indeces
            y_pred = y_pred_prob.argmax(axis=1)

            test_dict['predictions'] += y_pred.tolist()
            test_dict['labels'] += y.tolist()
            test_dict['correct'] += (y == y_pred).int().tolist()

    return pd.DataFrame.from_dict(test_dict)

