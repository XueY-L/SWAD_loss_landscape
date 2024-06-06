# KD3A: Unsupervised Multi-Source Decentralized Domain Adaptation via Knowledge Distillation

from os import path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append("..")

class DomainNetSet(Dataset):
    def __init__(self, data_paths, data_labels, transforms):
        super(DomainNetSet, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)


class DomainNetLoader:
    def __init__(
        self,
        domain_name='clipart',
        dataset_path=None,
        batch_size=64,
        num_workers=4,
        _C=None, 
    ):
        super(DomainNetLoader, self).__init__()
        self.domain_name = domain_name
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._C = _C
        # -------domainbed----------
        # https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py
        self.transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transforms_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def read_data(self, domain_name, split='train'):
        data_paths = []
        data_labels = []
        split_file = path.join(self.dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                data_path, label = line.split(' ')
                data_path = path.join(self.dataset_path, data_path)
                label = int(label)
                data_paths.append(data_path)
                data_labels.append(label)
        return data_paths, data_labels

    def get_dloader(self):
        '''
        return the ##whole## training/val/test dataloader of the target domain
        '''
        print(f'==> Loading DomainNet {self.domain_name}...')

        # dataset_path = path.join(base_path, 'dataset', 'DomainNet')
        train_data_path, train_data_label = self.read_data(self.domain_name, split="train")
        val_data_path, val_data_label = self.read_data(self.domain_name, split="val")
        test_data_path, test_data_label = self.read_data(self.domain_name, split="test")

        train_dataset = DomainNetSet(train_data_path, train_data_label, self.transforms_train)   # attention!!!
        val_dataset = DomainNetSet(val_data_path, val_data_label, self.transforms_test)
        test_dataset = DomainNetSet(test_data_path, test_data_label, self.transforms_test)  # attention!!!

        train_dloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)
        val_dloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)
        test_dloader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)
        
        print(f'Train sample number: {len(train_data_path)}, Val sample number: {len(val_data_path)}, Test sample number: {len(test_data_path)}')
        return train_dloader, val_dloader, test_dloader

    def get_source_dloaders(self, domain_ls):
        '''
            load source domains
            return train/val list, which length = len(source_domains), each element is a source dataloader
        '''
        print(f"==> Loading dataset {domain_ls}")
        train_loader_ls = []
        val_loader_ls = []
        test_loader_ls = []
        for d in domain_ls:
            # print(f'domain {d}')
            train_data_path, train_data_label = self.read_data(d, split="train")
            train_dataset = DomainNetSet(train_data_path, train_data_label, self.transforms_train,)
            train_dloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)
            train_loader_ls.append(train_dloader)

            val_data_path, val_data_label = self.read_data(d, split="val")
            val_dataset = DomainNetSet(val_data_path, val_data_label, self.transforms_test,)
            val_dloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)
            val_loader_ls.append(val_dloader)

            test_data_path, test_data_label = self.read_data(d, split="test")
            test_dataset = DomainNetSet(test_data_path, test_data_label, self.transforms_test,)
            test_dloader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)
            test_loader_ls.append(test_dloader)

            # print(len(train_dloader))
        return train_loader_ls, val_loader_ls, test_loader_ls
    
    def get_remix_dloader(self, domain_ls:list[str]=None, size:list[float]=None, split=[False,False,False]):
        '''
            Load target domains and mix them into one whole dataloader
            return test dataloader of target domains in yml file (remixed together)
        '''
        len_set = {
            'clipart':[26681,6844,14604],
            'infograph':[28678,7345,15582],
            'painting':[40196,10220,21850],
            'quickdraw':[96600,24150,51750],
            'real':[96589, 24317, 52041],
            'sketch':[38433,9779,20916],
        }
        print(domain_ls)
        train_data_path, train_data_label = [], []
        val_data_path, val_data_label = [], []
        test_data_path, test_data_label = [], []
        for idx, d in enumerate(domain_ls):
            tpath, tlabel = self.read_data(d, split="train")
            if split[0]:
                # random select
                select_order = np.random.choice(len(tpath), size=int(len_set[d][0]*size[idx]), replace=False)  # 无放回
                train_data_path.extend(tpath[i] for i in select_order)
                train_data_label.extend(tlabel[i] for i in select_order)
            else:
                train_data_path.extend(tpath)
                train_data_label.extend(tlabel)

            tpath, tlabel = self.read_data(d, split="val")
            if split[1]:
                # random select
                select_order = np.random.choice(len(tpath), size=int(len_set[d][1]*size[idx]), replace=False)  # 无放回
                val_data_label.extend(tpath[i] for i in select_order)
                val_data_label.extend(tlabel[i] for i in select_order)
            else:
                val_data_path.extend(tpath)
                val_data_label.extend(tlabel)

            tpath, tlabel = self.read_data(d, split="test")
            if split[2]:
                # random select
                select_order = np.random.choice(len(tpath), size=int(len_set[d][2]*size[idx]), replace=False)
                test_data_path.extend(tpath[i] for i in select_order)
                test_data_label.extend(tlabel[i] for i in select_order)
            else:
                test_data_path.extend(tpath)
                test_data_label.extend(tlabel)
        print(f'Train sample number: {len(train_data_path)}, Val sample number: {len(val_data_path)}, Test sample number: {len(test_data_path)}')

        train_dataset = DomainNetSet(train_data_path, train_data_label, self.transforms_train)
        train_dloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)
        val_dataset = DomainNetSet(val_data_path, val_data_label, self.transforms_test)
        val_dloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)
        test_dataset = DomainNetSet(test_data_path, test_data_label, self.transforms_test)
        test_dloader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)
        return train_dloader, val_dloader, test_dloader
    
    def get_remix_dloader_2(self, domain_ls:list[str]=None, size:list[int]=None):
        '''
        '''
        print(f'==> Loading DomainNet remix_dloader {domain_ls}...')
        train_loaders, test_loaders = [], []
        for idx, d in enumerate(domain_ls):
            train_data_path, train_data_label = [], []
            test_data_path, test_data_label = [], []

            tpath, tlabel = self.read_data(d, split="train")
            # random select
            select_order = np.random.choice(len(tpath), size=size[idx], replace=False)
            train_data_path.extend(tpath[i] for i in select_order)
            train_data_label.extend(tlabel[i] for i in select_order)

            tpath, tlabel = self.read_data(d, split="test")
            test_data_path.extend(tpath)
            test_data_label.extend(tlabel)
            print(f'Train sample number: {len(train_data_path)}\tTest sample number: {len(test_data_path)}')

            train_dataset = DomainNetSet(train_data_path, train_data_label, self.transforms_train)  # 暴搜用的是self.transforms_test
            train_dloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)
            test_dataset = DomainNetSet(test_data_path, test_data_label, self.transforms_test)
            test_dloader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)

            train_loaders.append(train_dloader)
            test_loaders.append(test_dloader)
        return train_loaders, test_loaders

    def get_cluster(self, path):
        f = open(path, 'r')
        lines = f.readlines()
        img_idx, labels = [], []
        for l in lines:
            t_loc = l.find('t')
            img_idx.append(int(l[:t_loc-1]))
            label = l[t_loc+8:-3].split(', ')
            label = [float(x) for x in label]
            labels.append(torch.tensor(label))
        # print(len(img_idx), len(labels))
        test_data_path, _ = self.read_data(self.domain_name, split="test")
        test_data_path = [test_data_path[index] for index in img_idx]
        test_dataset = DomainNetSet(test_data_path, labels, self.transforms_test)
        test_dloader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)
        return test_dloader
        