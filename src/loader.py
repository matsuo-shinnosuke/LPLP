import torch
import torchvision.transforms as transforms
import numpy as np

def set_loader(args, is_full_proportion=False):
    dataset_dir = args.dataset_dir/args.dataset / \
        str(args.num_instances)/str(args.kFold)
    ######### load data #######
    train_bags = np.load(dataset_dir/'train_bags.npy')
    train_labels = np.load(dataset_dir/'train_labels.npy')
    train_lps = np.load(dataset_dir/'train_partial_lps.npy')

    val_bags = np.load(dataset_dir/'val_bags.npy')
    val_labels = np.load(dataset_dir/'val_labels.npy')
    val_lps = np.load(dataset_dir/'val_partial_lps.npy')

    test_bags = np.load(dataset_dir/'test_bags.npy')
    test_labels = np.load(dataset_dir/'test_labels.npy')
    test_lps = np.load(dataset_dir/'test_partial_lps.npy')

    if is_full_proportion:
        train_lps = np.load(dataset_dir/'train_original_lps.npy')
        val_lps = np.load(dataset_dir/'val_original_lps.npy')
        test_lps = np.load(dataset_dir/'test_original_lps.npy')

    train_dataset = DatasetBag(
        data=train_bags, label=train_labels, lp=train_lps)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.mini_batch,
        shuffle=True,  num_workers=args.num_workers)

    val_dataset = DatasetBag(
        data=val_bags, label=val_labels, lp=val_lps)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.mini_batch,
        shuffle=False,  num_workers=args.num_workers)

    test_dataset = DatasetBag(
        data=test_bags, label=test_labels, lp=test_lps)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.mini_batch,
        shuffle=False,  num_workers=args.num_workers)

    return train_loader, val_loader, test_loader


class DatasetBag(torch.utils.data.Dataset):
    def __init__(self, data, label, lp):
        self.data = data
        self.label = label
        self.lp = lp

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),
                                 (0.2673, 0.2564, 0.2762))])
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        (b, w, h, c) = data.shape
        trans_data = torch.zeros((b, c, w, h))
        for i in range(b):
            trans_data[i] = self.transform(data[i])
        data = trans_data

        label = self.label[idx]
        label = torch.tensor(label).long()

        lp = self.lp[idx]
        lp = torch.tensor(lp).float()

        return data, label, lp
