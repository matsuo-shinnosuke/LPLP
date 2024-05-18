import numpy as np
import pickle
import sys
from pathlib import Path
from torchvision import datasets
from sklearn.model_selection import StratifiedKFold
import scipy.io as sio


from config import parse_option
from utils import fix_seed


def download_dataset(dataset='cifar10', data_folder='./dataset/'):
    if dataset == 'cifar10':
        datasets.CIFAR10(root=data_folder, download=True)
    elif dataset == 'svhn':
        datasets.SVHN(root=data_folder, split='train', download=True)
        datasets.SVHN(root=data_folder, split='test', download=True)
    else:
        raise ValueError(dataset)

def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()
    return data


def load_cifar10(dataset_dir='./dataset/'):
    X_train = None
    y_train = []

    for i in range(1, 6):
        data_dic = unpickle(
            dataset_dir/"cifar-10-batches-py/data_batch_{}".format(i))
        if i == 1:
            X_train = data_dic['data']
        else:
            X_train = np.vstack((X_train, data_dic['data']))
        y_train += data_dic['labels']

    test_data_dic = unpickle(dataset_dir/"cifar-10-batches-py/test_batch")
    X_test = test_data_dic['data']
    X_test = X_test.reshape(len(X_test), 3, 32, 32)
    y_test = np.array(test_data_dic['labels'])
    X_train = X_train.reshape((len(X_train), 3, 32, 32))
    y_train = np.array(y_train)

    train_img = X_train.transpose((0, 2, 3, 1))
    train_label = y_train
    test_img = X_test.transpose((0, 2, 3, 1))
    test_label = y_test

    return train_img, train_label, test_img, test_label

def load_svhn(dataset_dir='./dataset/'):
    train_data = sio.loadmat(f'{dataset_dir}/train_32x32.mat')
    x_train = train_data['X']
    x_train = x_train.transpose((3, 0, 1, 2))
    y_train = train_data['y'].reshape(-1)
    y_train[y_train == 10] = 0

    test_data = sio.loadmat(f'{dataset_dir}/test_32x32.mat')
    x_test = test_data['X']
    x_test = x_test.transpose((3, 0, 1, 2))
    y_test = test_data['y'].reshape(-1)
    y_test[y_test == 10] = 0

    return x_train, y_train, x_test, y_test

########### make bag ###########
def make_bag(dataset='cifar10', data_folder='./dataset/', num_instances=32, num_classes=10):
    if dataset == 'cifar10':
        data, label, test_data, test_label = load_cifar10(data_folder)
    elif dataset == 'svhn':
        data, label, test_data, test_label = load_svhn(data_folder)
    else:
        raise ValueError(dataset)

    # k-fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(skf.split(data, label)):
        train_data, train_label = data[train_idx], label[train_idx]
        val_data, val_label = data[val_idx], label[val_idx]

        output_path = Path(data_folder)/dataset/str(num_instances)/str(i)
        output_path.mkdir(parents=True, exist_ok=True)

        # train
        bags, labels, original_lps, partial_lps = get_bags(train_data, train_label,
                                                           num_classes=num_classes,
                                                           num_instances=num_instances,
                                                           num_posi_bags=400,
                                                           num_nega_bags=400)
        np.save('%s/train_bags' % (output_path), bags)
        np.save('%s/train_labels' % (output_path), labels)
        np.save('%s/train_original_lps' % (output_path), original_lps)
        np.save('%s/train_partial_lps' % (output_path), partial_lps)

        # val
        bags, labels, original_lps, partial_lps = get_bags(val_data, val_label,
                                                           num_classes=num_classes,
                                                           num_instances=num_instances,
                                                           num_posi_bags=100,
                                                           num_nega_bags=100)
        np.save('%s/val_bags' % (output_path), bags)
        np.save('%s/val_labels' % (output_path), labels)
        np.save('%s/val_original_lps' % (output_path), original_lps)
        np.save('%s/val_partial_lps' % (output_path), partial_lps)

        # test

        used_test_data, used_test_label = [], []
        for c in range(num_classes):
            used_test_data.extend(test_data[test_label == c])
            used_test_label.extend(test_label[test_label == c])
        test_data, test_label = np.array(
            used_test_data), np.array(used_test_label)

        bags, labels, original_lps, partial_lps = get_bags(test_data, test_label,
                                                           num_classes=num_classes,
                                                           num_instances=num_instances,
                                                           num_posi_bags=100,
                                                           num_nega_bags=100)
        np.save('%s/test_bags' % (output_path), bags)
        np.save('%s/test_labels' % (output_path), labels)
        np.save('%s/test_original_lps' % (output_path), original_lps)
        np.save('%s/test_partial_lps' % (output_path), partial_lps)


def get_label_proportion(num_bags=100, num_classes=10):
    proportion = np.random.rand(num_bags, num_classes)
    proportion /= proportion.sum(axis=1, keepdims=True)
    return proportion

def get_N_label_proportion(proportion, num_instances, num_classes):
    N = np.zeros(proportion.shape)
    for i in range(len(proportion)):
        p = proportion[i]
        for c in range(len(p)):
            if (c+1) != num_classes:
                num_c = int(np.round(num_instances*p[c]))
                if sum(N[i])+num_c >= num_instances:
                    num_c = int(num_instances-sum(N[i]))
            else:
                num_c = int(num_instances-sum(N[i]))

            N[i][c] = int(num_c)
    return N


def get_bags(data, label, num_classes, num_instances, num_posi_bags, num_nega_bags):
    # make poroportion
    proportion = get_label_proportion(num_posi_bags, num_classes)
    proportion_N = get_N_label_proportion(
        proportion, num_instances, num_classes)

    proportion_N_nega = np.zeros((num_nega_bags, num_classes))
    proportion_N_nega[:, 0] = num_instances

    proportion_N = np.concatenate([proportion_N, proportion_N_nega], axis=0)

    # make index
    idx = np.arange(len(label))
    idx_c = []
    for c in range(num_classes):
        x = idx[label[idx] == c]
        np.random.shuffle(x)
        idx_c.append(x)

    bags_idx = []
    for n in range(len(proportion_N)):
        bag_idx = []
        for c in range(num_classes):
            sample_c_index = np.random.choice(
                idx_c[c], size=int(proportion_N[n][c]), replace=False)
            bag_idx.extend(sample_c_index)

        np.random.shuffle(bag_idx)
        bags_idx.append(bag_idx)
    # bags_index.shape => (num_bags, num_instances)

    # make data, label, proportion
    bags, labels = data[bags_idx], label[bags_idx]
    original_lps = proportion_N / num_instances

    partial_lps = original_lps.copy()
    posi_nega = (original_lps[:, 0] != 1)
    partial_lps[posi_nega == 1, 0] = 0  # mask negative class
    partial_lps /= partial_lps.sum(axis=1, keepdims=True)  # normalize

    return bags, labels, original_lps, partial_lps

def main():
    args = parse_option()
    fix_seed(seed=args.seed)

    print('downloading dataset ...')
    download_dataset(dataset=args.dataset,
                     data_folder=args.dataset_dir)

    print('making bag ...')
    make_bag(dataset=args.dataset,
             data_folder=args.dataset_dir,
             num_instances=args.num_instances,
             num_classes=args.num_classes)

if __name__ == '__main__':
    main()
