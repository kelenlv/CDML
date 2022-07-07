import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
from pandas import DataFrame
# import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Subset
# import matplotlib
# matplotlib.use('TkAgg')
class Non_iid(Dataset): 
    def __init__(self, x, y):
        self.x_data = x.unsqueeze(1).to(torch.float32)
        self.y_data = y.to(torch.int64)
        self.batch_size = 32 # set batchsize in here
        self.cuda_available = torch.cuda.is_available()
        
    # Return the number of data 
    def __len__(self): 
        return len(self.x_data)
    
    # Sampling
    def __getitem__(self): 
        idx = np.random.randint(low = 0, high= len(self.x_data), size=self.batch_size) # random_index
        x = self.x_data[idx]
        y = self.y_data[idx]
        if self.cuda_available :
            return x.cuda(), y.cuda()
        else:
            return x, y

# Custom dataset prepration in Pytorch format
class SkinData(Dataset):
    def __init__(self, df, transform = None):
        
        self.df = df
        self.transform = transform
        self.targets = []
        
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, index):
        
        X = Image.open(self.df['path'][index]).resize((64, 64))
        y = torch.tensor(int(self.df['target'][index]))
        # img, target = self.data[index], self.targets[index]
        # print(y)
        self.targets.append(y)

        if self.transform:
            X = self.transform(X)
        
        return X, y
#=====================================================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
# IID HAM10000 datasets will be created based on this
def dataset_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users    
                          
def load_HAM10000(num_users, noniid):
    DIRICHLET_ALPHA = 0.5
    df = pd.read_csv('data/HAM10000_metadata.csv')
    # print(df.head())
    num_classes = 7
    lesion_type = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    # merging both folders of HAM1000 dataset -- part1 and part2 -- into a single directory
    imageid_path = {os.path.splitext(os.path.basename(x))[0]: x
                    for x in glob(os.path.join("data", '*', '*.jpg'))}


    #print("path---------------------------------------", imageid_path.get)
    df['path'] = df['image_id'].map(imageid_path.get)
    df['cell_type'] = df['dx'].map(lesion_type.get)
    df['target'] = pd.Categorical(df['cell_type']).codes
    # print(df['cell_type'].value_counts())
    # print(df['target'].value_counts())
    # Train-test split          
    train, test = train_test_split(df, test_size = 0.2)
    # print(test)
    train = train.reset_index()
    test = test.reset_index()
    print('-----------------------')
    # print(test)
    # Data preprocessing: Transformation 
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), 
                            transforms.RandomVerticalFlip(),
                            transforms.Pad(3),
                            transforms.RandomRotation(10),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean = mean, std = std)
                            ])
        
    test_transforms = transforms.Compose([
                            transforms.Pad(3),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean = mean, std = std)
                            ])    



    train_labels = []
    for i in range(len(train['target'])):
        train_labels.append(train['target'][i])

    train_labels = np.array(train_labels)
    print(train_labels)
        # With augmentation
    trainset = SkinData(train, transform = train_transforms)
    testset = SkinData(test, transform = test_transforms)

    if noniid == True:
        client_idcs = dirichlet_split_noniid(train_labels, alpha=DIRICHLET_ALPHA, n_clients=num_users)
        dict_users = [DataLoader(Subset(trainset, client_idcs[i]), batch_size=128, shuffle=True, num_workers =4,drop_last=True) for i in range(num_users)]
    else:
        dict_users = dataset_iid(trainset, num_users)

    dict_users_test = dataset_iid(testset, num_users)

    return trainset, testset, dict_users, dict_users_test#, client_idcs, train_labels

    

    # return dataset_train, dataset_test, dict_users, dict_users_test

def load_cifar10(num_users, noniid):
    DIRICHLET_ALPHA = 1
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    # if os.path.exists(path):
    #     trainset = torch.load(path)
    #     idx_train = range(len(trainset.targets))
    # else:
    trainset=datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    # idx_train = range(len(trainset.targets))
        # torch.save(trainset, path)

    
    testset = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(), normalize,]))

    train_labels = np.array(trainset.targets)
    print(train_labels)

    if noniid == True:
        client_idcs = dirichlet_split_noniid(train_labels, alpha=DIRICHLET_ALPHA, n_clients=num_users)
        dict_users = [DataLoader(Subset(trainset, client_idcs[i]), batch_size=128, shuffle=True, num_workers =4,drop_last=True) for i in range(num_users)]
        dict_users_test = dataset_iid(testset, num_users)
    else:
        dict_users = dataset_iid(trainset, num_users)
        dict_users_test = dataset_iid(testset, num_users)

    return trainset, testset, dict_users, dict_users_test

def load_cifar100(num_users, noniid):
    DIRICHLET_ALPHA = 0.5
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    trainset = datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        normalize,
                    ]), download=True)
    testset = datasets.CIFAR100(root='../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=True)
    train_labels = np.array(trainset.targets)
    

    if noniid == True:
        client_idcs = dirichlet_split_noniid(train_labels, alpha=DIRICHLET_ALPHA, n_clients=num_users)
        dict_users = [DataLoader(Subset(trainset, client_idcs[i]), batch_size=128, shuffle=True, drop_last=True, num_workers =4) for i in range(num_users)]
        dict_users_test = dataset_iid(testset, num_users)
    else:
        dict_users = dataset_iid(trainset, num_users)
        dict_users_test = dataset_iid(testset, num_users)

    return trainset, testset, dict_users, dict_users_test

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    参数为alpha的Dirichlet分布将数据索引划分为n_clients个子集
    '''
    n_classes = train_labels.max()+1 #10
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

    class_idcs = [np.argwhere(train_labels==y).flatten() 
           for y in range(n_classes)]
    # 记录每个K个类别对应的样本下标

    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

def dataset_noniid(iid_data, num_worker, num_classes): 
    alpha = 0.5 # if alpha is small --> increasing non-iidness, or if alpha is large --> increasing iidness
    for x, y in iid_data:
        idx   = [ torch.where(y == i) for i in range(num_classes) ] 
        # print(idx)
        data  = [ x[idx[i][0]] for i in range(num_classes) ] 
        label = [ torch.ones(len(data[i]))* i for i in range(num_classes)]

    s = np.random.dirichlet(np.ones(num_classes)*alpha, num_worker)
    data_dist = np.zeros((num_worker,num_classes))

    for j in range(num_worker):
        data_dist[j] = ((s[j]*len(data[0])).astype('int') / (s[j]*len(data[0])).astype('int').sum() * len(data[0])).astype('int')
        data_num     = data_dist[j].sum()
        data_dist[j][np.random.randint(low=0,high=num_classes)] += ((len(data[0]) - data_num) )
        data_dist    = data_dist.astype('int')
        
    X = []
    Y = []
    for j in range(num_worker):
        x_data = []
        y_data = []
        for i in range(num_classes):
            if data_dist[j][i] != 0:
                d_index = np.random.randint(low=0, high=len(data[i]), size=data_dist[j][i])
                x_data.append(data[i][d_index])
                y_data.append(label[i][d_index])
        X.append(torch.cat(x_data))
        Y.append(torch.cat(y_data))

    Non_iid_dataset  = [Non_iid(X[j],Y[j]) for j in range(num_worker)]
    return Non_iid_dataset

if __name__ == "__main__":
    N_CLIENTS = 20
    DIRICHLET_ALPHA = 0.5

    # train_data = datasets.EMNIST(root=".", split="byclass", download=True, train=True)
    # test_data = datasets.EMNIST(root=".", split="byclass", download=True, train=False)
    noniid=True


    train_data, testset, dict_users, dict_users_test, client_idcs, train_labels = load_HAM10000(N_CLIENTS, noniid)#cifar10

    # input_sz, num_cls = train_data.data[0].shape[0],  len(train_data.classes)
    num_cls = 7
    # print(train_data.targets)
    # train_labels = np.array(train_data.targets)

    # # 我们让每个client不同label的样本数量不同，以此做到Non-IID划分
    # client_idcs = dirichlet_split_noniid(train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)

    # dict_users = [DataLoader(Subset(train_data, client_idcs[i]), batch_size=128, shuffle=True, num_workers =4) for i in range(N_CLIENTS)]

    

    # print(client_idcs)
    # 展示不同client的不同label的数据分布
    print('begin plot')
    plt.figure(figsize=(20,3))
    plt.hist([train_labels[idc]for idc in client_idcs], stacked=True, 
            bins=np.arange(min(train_labels)-0.5, max(train_labels) + 1.5, 1),
            label=["Client {}".format(i) for i in range(N_CLIENTS)], rwidth=0.5)
    plt.xticks(np.arange(num_cls), train_data.classes)
    plt.legend()
    # plt.show()
    plt.savefig('plot.png', bbox_inches='tight')