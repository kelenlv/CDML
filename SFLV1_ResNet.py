#============================================================================
# SplitfedV1 (SFLV1) learning: ResNet18 on HAM10000
# HAM10000 dataset: Tschandl, P.: The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions (2018), doi:10.7910/DVN/DBW86T

# We have three versions of our implementations
# Version1: without using socket and no DP+PixelDP
# Version2: with using socket but no DP+PixelDP
# Version3: without using socket but with DP+PixelDP

# This program is Version1: Single program simulation 
# ============================================================================
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
from datasets import load_HAM10000, load_cifar10, load_cifar100
import random
import numpy as np
import os
import time 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
from thop import profile #modified by lkx
from thop import clever_format

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    

#===================================================================
program = "SFLV1 ResNet18 on noniid CIFAR10"
print(f"---------{program}----------")              # this is to identify the program in the slurm outputs files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))     

#===================================================================
# No. of users
num_users = 20
epochs = 100
frac = 1        # participation of clients; if 1 then 100% clients participate in SFLV1
lr = 0.0001


#=====================================================================================================
#                           Client-side Model definition
#=====================================================================================================
# Model at client side
class ResNet18_client_side(nn.Module):
    def __init__(self):
        super(ResNet18_client_side, self).__init__()
        self.layer1 = nn.Sequential (
                nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 3, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.MaxPool2d(kernel_size = 3, stride = 2, padding =1),
            )
        self.layer2 = nn.Sequential  (
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64),              
            )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        
    def forward(self, x):
        resudial1 = F.relu(self.layer1(x))
        out1 = self.layer2(resudial1)
        out1 = out1 + resudial1 # adding the resudial inputs -- downsampling not required in this layer
        resudial2 = F.relu(out1)
        return resudial2
 
 
           

net_glob_client = ResNet18_client_side()
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_client = nn.DataParallel(net_glob_client)    

net_glob_client.to(device)
print(net_glob_client) 
  

#=====================================================================================================
#                           Server-side Model definition
#=====================================================================================================
# Model at server side
class Baseblock(nn.Module):
    expansion = 1
    def __init__(self, input_planes, planes, stride = 1, dim_change = None):
        super(Baseblock, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, planes, stride =  stride, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride = 1, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dim_change = dim_change
        
    def forward(self, x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        
        if self.dim_change is not None:
            res =self.dim_change(res)
            
        output += res
        output = F.relu(output)
        
        return output


class ResNet18_server_side(nn.Module):
    def __init__(self, block, num_layers, classes):
        super(ResNet18_server_side, self).__init__()
        self.input_planes = 64
        self.layer3 = nn.Sequential (
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64),       
                )   
        
        self.layer4 = self._layer(block, 128, num_layers[0], stride = 2)
        self.layer5 = self._layer(block, 256, num_layers[1], stride = 2)
        self.layer6 = self._layer(block, 512, num_layers[2], stride = 2)
        
        self.averagePool = nn.AdaptiveAvgPool2d((1, 1))
        # self. averagePool = nn.AvgPool2d(kernel_size = 1, stride = 1, padding = 1)#modified by lkx (2,7) -> (2,1) -> (1,1)
        self.fc = nn.Linear(512 * block.expansion, classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        
    def _layer(self, block, planes, num_layers, stride = 2):
        dim_change = None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = nn.Sequential(nn.Conv2d(self.input_planes, planes*block.expansion, kernel_size = 1, stride = stride),
                                       nn.BatchNorm2d(planes*block.expansion))
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride = stride, dim_change = dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion
            
        return nn.Sequential(*netLayers)
        
    
    def forward(self, x):
        out2 = self.layer3(x)
        out2 = out2 + x          # adding the resudial inputs -- downsampling not required in this layer
        x3 = F.relu(out2)
        
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)

        x7 = self.averagePool(x6)#modified by lkx
        # x7 = F.avg_pool2d(x6,7)#modified by lkx
        x8 = x7.view(x7.size(0), -1) 
        y_hat =self.fc(x8)
        
        return y_hat

#modified by lkx
noniid = True# False
dataset_train, dataset_test, dict_users, dict_users_test = load_HAM10000(num_users, noniid)#load_cifar10
cc =  7 # 7

net_glob_server = ResNet18_server_side(Baseblock, [2,2,2], cc) #cc is my numbr of classes #modified by lkx
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_server = nn.DataParallel(net_glob_server)   # to use the multiple GPUs 

net_glob_server.to(device)
print(net_glob_server)      

#===================================================================================
# For Server Side Loss and Accuracy 
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []


criterion = nn.CrossEntropyLoss()
count1 = 0
count2 = 0
#====================================================================================================
#                                  Server Side Program
#====================================================================================================
# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

# to print train - test together in each round-- these are made global
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []
time_server = []
time_client = []

w_glob_server = net_glob_server.state_dict()
w_locals_server = []

#client idx collector
idx_collect = []
l_epoch_check = False
fed_check = False
# Initialization of net_model_server and net_server (server-side model)
net_model_server = [net_glob_server for i in range(num_users)]
net_server = copy.deepcopy(net_model_server[0]).to(device)
#optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)

# Server-side function associated with Training 
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    global net_model_server, criterion, optimizer_server, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_server, w_glob_server, net_server
    global loss_train_collect_user, acc_train_collect_user, lr
    # flops, params = profile(net_server, inputs=(fx_client, ))#modofied by lkx
    # flops, params = clever_format([flops, params], "%.3f")
    # print('#calculating params of server:',flops, params)
    begin =  time.perf_counter() 
    net_server = copy.deepcopy(net_model_server[idx]).to(device)
    net_server.train()
    optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)
    
    # with torch.autograd.profiler.profile(enabled=True) as prof:

    # train and update
    optimizer_server.zero_grad()
    
    fx_client = fx_client.to(device)
    y = y.to(device)
    
    #---------forward prop-------------
    fx_server = net_server(fx_client)
    
    # calculate loss
    loss = criterion(fx_server, y)
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)
    
    #--------backward prop--------------
    loss.backward()
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()
    
    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())
    
    # Update the server-side model for the current batch
    net_model_server[idx] = copy.deepcopy(net_server)

    end =  time.perf_counter() 

    time_server.append(end-begin)
    # print(prof.key_averages().table(sort_by="self_cuda_time_total"))#self_cpu_time_total

    # count1: to track the completion of the local batch associated with one client
    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train)/len(batch_acc_train)           # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train)/len(batch_loss_train)
        
        batch_acc_train = []
        batch_loss_train = []
        count1 = 0
        
        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train, loss_avg_train))
        
        # copy the last trained model in the batch       
        w_server = net_server.state_dict()      
        
        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch-1:
            
            l_epoch_check = True                # to evaluate_server function - to check local epoch has completed or not 
            # We store the state of the net_glob_server() 
            w_locals_server.append(copy.deepcopy(w_server))
            
            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)
            
            #print("accuracy = ", acc_avg_train)
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train
                        
            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)
            
            # collect the id of each new user                        
            if idx not in idx_collect:
                idx_collect.append(idx) 
                #print(idx_collect)
        
        # This is for federation process--------------------
        if len(idx_collect) == num_users:
            fed_check = True                                                  # to evaluate_server function  - to check fed check has hitted
            # Federation process at Server-Side------------------------- output print and update is done in evaluate_server()
            # for nicer display 
                                   
            w_glob_server = FedAvg(w_locals_server)   
            
            # server-side global model update and distribute that model to all clients ------------------------------
            net_glob_server.load_state_dict(w_glob_server)    
            net_model_server = [net_glob_server for i in range(num_users)]
            
            w_locals_server = []
            idx_collect = []
            
            acc_avg_all_user_train = sum(acc_train_collect_user)/len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user)/len(loss_train_collect_user)
            
            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)
            
            acc_train_collect_user = []
            loss_train_collect_user = []
            
    # send gradients to the client               
    return dfx_client

# Server-side functions associated with Testing
def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_model_server, criterion, batch_acc_test, batch_loss_test, check_fed, net_server, net_glob_server 
    global loss_test_collect, acc_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, w_glob_server, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train
    
    net = copy.deepcopy(net_model_server[idx]).to(device)
    net.eval()
  
    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device) 
        #---------forward prop-------------
        fx_server = net(fx_client)
        
        # calculate loss
        loss = criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)
        
        
        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())
        
               
        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test)/len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test)/len(batch_loss_test)
            
            batch_acc_test = []
            batch_loss_test = []
            count2 = 0
            
            prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test, loss_avg_test))
            
            # if a local epoch is completed   
            if l_epoch_check:
                l_epoch_check = False
                
                # Store the last accuracy and loss
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test
                        
                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)
                
            # if federation is happened----------                    
            if fed_check:
                fed_check = False
                print("------------------------------------------------")
                print("------ Federation process at Server-Side ------- ")
                print("------------------------------------------------")
                
                acc_avg_all_user = sum(acc_test_collect_user)/len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user)/len(loss_test_collect_user)
            
                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user= []
                              
                print("====================== SERVER V1==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train, loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user, loss_avg_all_user))
                print("==========================================================")
         
    return 

#==============================================================================================================
#                                       Clients-side Program
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
class Client(object):
    def __init__(self, net_client_model, idx, lr, device, noniid, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        #self.selected_clients = []
        if noniid == True:
            self.ldr_train = idxs
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size = 256, shuffle = True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size = 256, shuffle = True)
        

    def train(self, net):
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr = self.lr) 
        
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)

                # flops, params = profile(net, inputs=(images, ))#modified by lkx
                # flops, params = clever_format([flops, params], "%.3f")
                # print('#Calculating of params in client:', flops, params)

                optimizer_client.zero_grad()
                #---------forward prop-------------
                print(torch.cuda.memory_allocated())
                fx = net(images)
                print(torch.cuda.memory_allocated())
                print('one-step')
                client_fx = fx.clone().detach().requires_grad_(True)
                
                

                # Sending activations to server and receiving gradients from server
                dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)
                
                #--------backward prop -------------
                fx.backward(dfx)
                optimizer_client.step()
                            
            
            #prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))
           
        return net.state_dict() 
    
    def evaluate(self, net, ell):
        net.eval()
           
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx = net(images)
                
                # Sending activations to server 
                evaluate_server(fx, labels, self.idx, len_batch, ell)
            
            #prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))
            
        return          


#------------ Training And Testing  -----------------
net_glob_client.train()
#copy weights
w_glob_client = net_glob_client.state_dict()

time_client = {}
m = max(int(frac * num_users), 1)
idxs_users = np.random.choice(range(num_users), m, replace = False)
# print(idxs_users)
for idx in idxs_users:    
    time_client[idx] = []
# Federation takes place after certain local epochs in train() client-side
# this epoch is global epoch, also known as rounds
for iter in range(epochs):
    w_locals_client = []
      
    for idx in idxs_users:
        local = Client(net_glob_client, idx, lr, device, noniid, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx])
        # Training ------------------
        begin =  time.perf_counter() 
        w_client = local.train(net = copy.deepcopy(net_glob_client).to(device))
        w_locals_client.append(copy.deepcopy(w_client))
        end =  time.perf_counter() 
        time_client[idx].append(end-begin)
        # print(time_client)
        # Testing -------------------
        local.evaluate(net = copy.deepcopy(net_glob_client).to(device), ell= iter)
        
            
    # Ater serving all clients for its local epochs------------
    # Fed  Server: Federation process at Client-Side-----------
    print("-----------------------------------------------------------")
    print("------ FedServer: Federation process at Client-Side ------- ")
    print("-----------------------------------------------------------")
    w_glob_client = FedAvg(w_locals_client)   

    # Update client-side global model 
    net_glob_client.load_state_dict(w_glob_client)    
    
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

print('training time in Server is ' + str(sum(time_server)))
print('finally:', avg_time_client, std_time_client )

# round_process = [i for i in range(1, len(acc_train_collect)+1)]
# df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect, 
#                 'avg_time_client':avg_time_client, 'std_time_client': std_time_client, 'time_server':sum(time_server) })     
# file_name = program+".xlsx"    
# df.to_excel(file_name, sheet_name= "v1_test", index = False)    


#=============================================================================
#                         Program Completed
#=============================================================================