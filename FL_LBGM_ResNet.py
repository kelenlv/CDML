#===========================================================
# Federated learning: ResNet18 on HAM10000
# HAM10000 dataset: Tschandl, P.: The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions (2018), doi:10.7910/DVN/DBW86T

# We have three versions of our implementations
# Version1: without using socket and no DP+PixelDP
# Version2: with using socket but no DP+PixelDP
# Version3: without using socket but with DP+PixelDP

# This program is Version1: Single program simulation 
# ===========================================================
from pickle import FALSE
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

import math
import random
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import copy
import time 
from datasets import load_HAM10000, load_cifar10, load_cifar100
# from thop import profile #modified by lkx
# from thop import clever_format

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    


#===================================================================  
program = "FL ResNet18 on iid CIFAR10"
print(f"---------{program}----------")              # this is to identify the program in the slurm outputs files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# To print in color during test/train 
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))    

def get_model_grads(model, scaling_factor=1, flatten=False, numpy=False):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.clone()*scaling_factor)

    if flatten:
        grads = [_.flatten().cpu().numpy() if numpy else _.flatten()
                 for _ in grads]
    return grads

def calc_projection(a, b, device):
    # projection of vector a on vector b
    a = a.clone().to(device)
    b = b.clone().to(device)
    proj = (torch.dot(a, b) / torch.dot(b, b)).item()
    # print(type(proj))#<class 'float'>
    # error of projection calculation
    cos_alpha = torch.nn.CosineSimilarity(dim=0, eps=1e-6)(a, b)
    sin2_alpha = 1 - cos_alpha**2
    try:
        assert sin2_alpha >= 0.0 and sin2_alpha <= 1.0
    except Exception as e:
        if torch.isclose(sin2_alpha, torch.tensor(0.0), atol=1e-6):
            sin2_alpha = 0.0
        else:
            print(f'value of sin2_alpha: {sin2_alpha}')
            print(e)
            exit()

    return proj, sin2_alpha

def add_param_list(param1, param2):
    if not param1:
        return param2

    assert len(param1) == len(param2)
    for idx in range(len(param1)):
        param1[idx] = param1[idx].cpu() + param2[idx].cpu()

    return param1

def model_update(model, grads, args, device, prev_v=[], epoch=0):
    idx = 0
    accum_v = []
    for param in model.parameters():
        d = grads[idx]

        if args.momentum:
            if len(prev_v):
                d = args.momentum * prev_v[idx] + d
                accum_v.append(d)
            else:
                accum_v.append(d)

        with torch.no_grad():
            lr = args.lr if not args.scheduler else get_scheduled_lr(
                args, epoch)
            param.copy_(param.add(-args.lr, d.to(device)))
        idx += 1
    return accum_v
#===================================================================
# No. of users
num_users = 20
epochs = 300#
frac = 1
lr = 0.0001

#==============================================================================================================
#                                  Client Side Program 
#==============================================================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

# Client-side functions associated with Training and Testing
class LocalUpdate(object):
    def __init__(self, idx, lr, device, noniid, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.error_tol = 0.2 #type=float
        self.residual = False#type=booltype
        self.local_num_samples = len(idxs)
        if noniid == True:
            self.ldr_train = idxs
            print('in noniid')
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size = 256, shuffle = True, drop_last=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size = 256, shuffle = True, drop_last=True)
        

    def lbgm(self, node_model, sdirs, worker_residuals):
        if self.residual:
            sdirs, worker_residuals, avg_rho = self.lbgm_approximation(node_model, sdirs, worker_residuals)
        else:
            sdirs, _,  avg_rho = self.lbgm_approximation( node_model, sdirs, [])
            worker_residuals = []

        return sdirs, worker_residuals, avg_rho

    def lbgm_approximation(self, model, lbgs, residuals):
        accum_rho = 0.0
        accum_lbgs = []
        accum_residuals = []
        # uplink = 0
        for i, p in enumerate(model.parameters()):
            size = p.grad.size()
            grad_flat = p.grad.clone().flatten()
            grad_res = residuals[i] if len(
                residuals) else torch.zeros_like(grad_flat)
            grad_flat = grad_flat + grad_res

            if len(lbgs):
                rho, error = calc_projection(
                    grad_flat, lbgs[i], self.device)
            else:
                error = 1.0

            if error <= self.error_tol:
                update = rho * lbgs[i]
                accum_residuals.append(grad_flat - update)
                accum_rho += rho
                accum_lbgs.append(lbgs[i])
                # uplink += 1
            else:
                update = grad_flat
                accum_lbgs.append(update.clone())
                accum_residuals.append(torch.zeros_like(update))
                # uplink += len(update)
            p.grad.copy_(update.reshape(size))

        # number of layers = (i+1) and not i because its zero-indexed
        return accum_lbgs, accum_residuals,  accum_rho / (i + 1)#uplink,

    def train(self,  sdirs, worker_residuals, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr = self.lr, momentum = 0.9)
        # optimizer = torch.optim.Adam(net.parameters(), lr = self.lr)
        
        epoch_acc = []
        epoch_loss = []
        for iter in range(self.local_ep):
            batch_acc = []
            batch_loss = []
            
            for batch_idx, (images, labels) in enumerate(self.ldr_train):#
                images = images.to(self.device)
                labels = labels.to(self.device)
                # flops, params = profile(net, inputs=(images, ))#modified by lkx
                # flops, params = clever_format([flops, params], "%.3f")
                # print('#Calculating of params in client:', flops, params)

                optimizer.zero_grad()
                #---------forward prop-------------
                # print(torch.cuda.memory_allocated())
                fx = net(images)
                # print(torch.cuda.memory_allocated())
                # print('one-step')
                # calculate loss
                loss = self.loss_func(fx, labels)
                # calculate accuracy
                acc = calculate_accuracy(fx, labels)
                
                #--------backward prop--------------
                loss.backward()
                optimizer.step()
                              
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())

                # worker_grads = get_model_grads(net)
                assert batch_idx < 2 - 1 #args.tau
                break
            
            error = 0
            sdirs, worker_residuals, avg_rho = self.lbgm(net, sdirs, worker_residuals)
            error = 1 - avg_rho
            worker_grads = get_model_grads(net)
            worker_grads = [_ * self.local_num_samples/50000
                    for _ in worker_grads]

            prRed('Client{} Train => Local Epoch: {}  \tAcc: {:.3f} \tLoss: {:.4f}'.format(self.idx,
                        iter, acc.item(), loss.item()))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(sum(batch_acc)/len(batch_acc))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc),\
             worker_residuals, worker_grads, error, sdirs
    
    def evaluate(self, net):
        net.eval()
           
        epoch_acc = []
        epoch_loss = []
        with torch.no_grad():
            batch_acc = []
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                # images = input
                # labels = targets
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx = net(images)
                
                # calculate loss
                loss = self.loss_func(fx, labels)
                # calculate accuracy
                acc = calculate_accuracy(fx, labels)
                                 
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
            
            prGreen('Client{} Test =>                     \tLoss: {:.4f} \tAcc: {:.3f}'.format(self.idx, loss.item(), acc.item()))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(sum(batch_acc)/len(batch_acc))
        return sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)


#modified by lkx
noniid = False #True
dataset_train, dataset_test, dict_users, dict_users_test = load_cifar10(num_users, noniid)#0load_HAM10000
cc =  10 #  100 7

#====================================================================================================
#                               Server Side Program
#====================================================================================================
def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

#=============================================================================
#                    Model definition: ResNet18
#============================================================================= 
# building a ResNet18 Architecture
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet18(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.averagePool = nn.AdaptiveAvgPool2d((1, 1)) #modified by lkx
        # self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        x = self.averagePool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



net_glob = ResNet18(BasicBlock, [2, 2, 2, 2], cc) #7 is my numbr of classes

# input = torch.randn(1, 3, 224, 224)
# flops, params = profile(net_glob, inputs=(input, ))
# flops, params = clever_format([flops, params], "%.3f")
# print('#Calculating of params in client:', flops, params)

if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob = nn.DataParallel(net_glob)   # to use the multiple GPUs 

net_glob.to(device)
print(net_glob)      

#===========================================================================================
# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
#====================================================
net_glob.train()
# copy weights

w_glob = net_glob.state_dict()

time_client = {}
m = max(int(frac * num_users), 1)
idxs_users = np.random.choice(range(num_users), m, replace = False)
# print(idxs_users)
for idx in idxs_users:    
    time_client[idx] = []
    
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []

worker_residuals = {}
worker_sdirs = {}

for iter in range(epochs):
    w_locals, loss_locals_train, acc_locals_train, loss_locals_test, acc_locals_test = [], [], [], [], []
    # grad_agg = []
    # m = max(int(frac * num_users), 1)
    # idxs_users = np.random.choice(range(num_users), m, replace = False)
    worker_grad_sum = 0
    # Training/Testing simulation
    for idx in idxs_users: # each client
        local = LocalUpdate(idx, lr, device, noniid, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx])
        # Training ------------------
        begin =  time.perf_counter() 
        w, loss_train, acc_train,\
             worker_residuals, worker_grads, error_per_worker, worker_sdirs = local.train(worker_sdirs, worker_residuals, net = copy.deepcopy(net_glob).to(device))
        end =  time.perf_counter() 
        # avg_error += error_per_worker
        time_client[idx].append(end-begin)
        
        worker_grad_sum = add_param_list(worker_grad_sum, worker_grads)
        # print(w_locals[0].size())
        # print(w_locals[0].dtype)
        w_locals.append(copy.deepcopy(w))
        loss_locals_train.append(copy.deepcopy(loss_train))
        acc_locals_train.append(copy.deepcopy(acc_train))
        # grad_agg.append(copy.deepcopy(worker_grad_sum))
        # worker_sdirs[w] = tmp_sdir
        # Testing -------------------
        loss_test, acc_test = local.evaluate(net = copy.deepcopy(net_glob).to(device))
        loss_locals_test.append(copy.deepcopy(loss_test))
        acc_locals_test.append(copy.deepcopy(acc_test))
        
        
    
    # Federation process
    w_glob = FedAvg(w_locals)
    

    print("------------------------------------------------")
    print("------ Federation process at Server-Side -------")
    print("------------------------------------------------")
    # update global model --- copy weight to net_glob -- distributed the model to all users
    
    net_glob.load_state_dict(w_glob)
    
    # Train/Test accuracy
    acc_avg_train = sum(acc_locals_train) / len(acc_locals_train)
    acc_train_collect.append(acc_avg_train)
    acc_avg_test = sum(acc_locals_test) / len(acc_locals_test)
    acc_test_collect.append(acc_avg_test)
    
    # Train/Test loss
    loss_avg_train = sum(loss_locals_train) / len(loss_locals_train)
    loss_train_collect.append(loss_avg_train)
    loss_avg_test = sum(loss_locals_test) / len(loss_locals_test)
    loss_test_collect.append(loss_avg_test)
    
    
    print('------------------- SERVER ----------------------------------------------')
    print('Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(iter, acc_avg_train, loss_avg_train))
    print('Test:  Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(iter, acc_avg_test, loss_avg_test))
    print('-------------------------------------------------------------------------')
   

#===================================================================================     

print("Training and Evaluation completed!")    

#===============================================================================
# Save output data to .excel file (we use for comparision plots)
avg_time_client = []
for idx in idxs_users:
    # print(idx)
    print('training time in Client '+ str(idx) + ' is ' + str(sum(time_client[idx])))
    avg_time_client.append(sum(time_client[idx]))
std_time_client = np.std(avg_time_client)
avg_time_client = np.mean(avg_time_client)

print('finally:', avg_time_client, std_time_client)
# print('training time in Server is ' + str(sum(time_server)))

# round_process = [i for i in range(1, len(acc_train_collect)+1)]
# df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect,
#                 'avg_time_client':avg_time_client, 'std_time_client': std_time_client})     
# file_name = program+".xlsx"    
# df.to_excel(file_name, sheet_name= "v1_test", index = False)     

#=============================================================================
#                         Program Completed
#============================================================================= 





