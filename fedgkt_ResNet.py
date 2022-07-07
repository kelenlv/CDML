import torch
import argparse
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
from datasets import load_HAM10000, load_cifar10,  load_cifar100
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
from utils import *
from  GKTServerTrainer import *

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    

#===================================================================
program = "FedGKT ResNet18 on iid CIFAR10"
print(f"---------{program}----------")              # this is to identify the program in the slurm outputs files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))     

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    # parser.add_argument('--model_client', type=str, default='resnet5', metavar='N',
    #                     help='neural network used in training')

    # parser.add_argument('--model_server', type=str, default='resnet32', metavar='N',
    #                     help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')

    parser.add_argument('--partition_method', type=str, default='homo', metavar='N',
                        help='how to partition the dataset on local workers')

    # parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
    #                     help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=5e-4)

    # parser.add_argument('--epochs_client', type=int, default=3, metavar='EP',
    #                     help='how many epochs will be trained locally')

    # parser.add_argument('--local_points', type=int, default=5000, metavar='LP',
    #                     help='the approximate fixed number of data points we will have on each local worker')

    # parser.add_argument('--client_number', type=int, default=4, metavar='NN',
    #                     help='number of workers in a distributed cluster')

    # parser.add_argument('--comm_round', type=int, default=300,
    #                     help='how many round of communications we shoud use')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    # parser.add_argument('--loss_scale', type=float, default=1024,
    #                     help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--no_bn_wd', action='store_true', help='Remove batch norm from weight decay')

    # knowledge distillation
    parser.add_argument('--temperature', default=3.0, type=float, help='Input the temperature: default(3.0)')
    # parser.add_argument('--epochs_server', type=int, default=3, metavar='EP',
    #                     help='how many epochs will be trained on the server side')
    parser.add_argument('--alpha', default=1.0, type=float, help='Input the relative weight: default(1.0)')
    parser.add_argument('--optimizer', default="SGD", type=str, help='optimizer: SGD, Adam, etc.')
    parser.add_argument('--whether_training_on_client', default=1, type=int)
    parser.add_argument('--whether_distill_on_the_server', default=0, type=int)
    # parser.add_argument('--client_model', default="resnet4", type=str)
    parser.add_argument('--weight_init_model', default="resnet32", type=str)
    parser.add_argument('--running_name', default="default", type=str)
    parser.add_argument('--sweep', default=1, type=int)
    parser.add_argument('--multi_gpu_server', action='store_true')
    parser.add_argument('--test', action='store_true',
                        help='test mode, only run 1-2 epochs to test the bug of the program')

    parser.add_argument('--gpu_num_per_server', type=int, default=8,
                        help='gpu_num_per_server')

    args = parser.parse_args()
    return args
#===================================================================
parser = argparse.ArgumentParser()
args = add_args(parser)
# No. of users
num_users = 20
epochs = 100
frac = 1        # participation of clients; if 1 then 100% clients participate in SFLV1
lr = 0.001

#modified by lkx
noniid =False #True
dataset_train, dataset_test, dict_users, dict_users_test = load_HAM10000(num_users, noniid)#load_cifar100
cc = 7 # 10

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
#=====================================================================================================
#                           Client-side Model definition
#=====================================================================================================
# Model at client side
class ResNet18_client_side(nn.Module):
    def __init__(self, block, num_classes):
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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

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
        extracted_features = resudial2
        
        x = self.avgpool(resudial2)  # B x 64 x 1 x 1
        x_f = x.view(x.size(0), -1)  # B x 64
        logits = self.fc(x_f)  # B x num_classes
        return logits, extracted_features
 
           
net_glob_client = ResNet18_client_side(Baseblock, cc)
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_client = nn.DataParallel(net_glob_client)    

net_glob_client.to(device)
print(net_glob_client) 
  

#=====================================================================================================
#                           Server-side Model definition
#=====================================================================================================
# Model at server side



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
criterion_KL = utils.KL_Loss(3.0)#self.args.temperature
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


# def calculate_accuracy(fx, y):
#     preds = fx.max(1, keepdim=True)[1]
#     correct = preds.eq(y.view_as(preds)).sum()
#     acc = 100.00 *correct.float()/preds.shape[0]
#     return acc

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
        # self.train_and_eval(round_idx, self.args.epochs_server)
        # self.scheduler.step(self.best_acc, epoch=round_idx)



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
    def __init__(self, server_logits_dict, net_client_model, idx, lr, device, noniid, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
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
            # key: batch_index; value: extracted_feature_map
        extracted_feature_dict = dict()

        # key: batch_index; value: logits
        logits_dict = dict()

        # key: batch_index; value: label
        labels_dict = dict()

        optimizer_client = torch.optim.Adam(net.parameters(), lr = self.lr) 
        net.train()
        epoch_loss = []    
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                # flops, params = profile(net, inputs=(images, ))#modified by lkx
                # flops, params = clever_format([flops, params], "%.3f")
                # print('#Calculating of params in client:', flops, params)
                #---------forward prop-------------
                log_probs, _ = net(images)
                # client_fx = fx.clone().detach().requires_grad_(True)

                # if self.args.whether_training_on_client == 1:
                loss_true = criterion(log_probs, labels)
                # if len(server_logits_dict) != 0:
                if server_logits_dict[idx].size > 0:
                    large_model_logits = torch.from_numpy(server_logits_dict[batch_idx]).to(
                        self.device)
                    # print(log_probs.size())
                    # print(large_model_logits.size())
                    loss_kd = criterion_KL(log_probs, large_model_logits)
                    loss = loss_true + 0.5 * loss_kd
                else:
                    loss = loss_true

                optimizer_client.zero_grad()
                loss.backward()
                optimizer_client.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))                
            # client_logits_dict[idx] = logits_dict 
            #prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))

        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.device), labels.to(self.device)

            # logging.info("shape = " + str(images.shape))
            log_probs, extracted_features = net(images)
            print(extracted_features.size())#torch.Size([256, 64, 9, 9])
            print(extracted_features.dtype)#torch.float32
            # print(log_probs.size())#torch.Size([256, 100])
            # print(log_probs.dtype)#torch.float32
            extracted_feature_dict[batch_idx] = extracted_features.cpu().detach().numpy()           
            log_probs = log_probs.cpu().detach().numpy()
           
            logits_dict[batch_idx] = log_probs
            labels_dict[batch_idx] = labels.cpu().detach().numpy()   

        return extracted_feature_dict, logits_dict, labels_dict
    
    def evaluate(self, net, ell):
    # for test - key: batch_index; value: extracted_feature_map
        extracted_feature_dict_test = dict()
        labels_dict_test = dict()
        net.eval()
           
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                test_images, test_labels = images.to(self.device), labels.to(self.device)
                _, extracted_features_test = net(test_images)
                extracted_feature_dict_test[batch_idx] = extracted_features_test.cpu().detach().numpy()
                labels_dict_test[batch_idx] = test_labels.cpu().detach().numpy()
            
            #prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))
            
        return extracted_feature_dict_test, labels_dict_test   


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

server_trainer = GKTServerTrainer(m, device, net_server, args)
server_logits_dict = dict()  
# client_logits_dict = dict()
# for idx in idxs_users:
#     server_logits_dict[idx] = []

for iter in range(epochs): #global communication
    # w_locals_client = []
    extracted_feature_dict = dict()
    logits_dict = dict()
    labels_dict = dict()
    extracted_feature_dict_test = dict()
    labels_dict_test = dict() 

    server_logits_dict = dict()      
    # client_logits_dict = dict()
    for idx in idxs_users:
         server_logits_dict[idx]= np.array([])

    for idx in idxs_users:
        local = Client(server_logits_dict[idx], net_glob_client, idx, lr, device, noniid, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx])
        # Training ------------------
        begin =  time.perf_counter() 
        extracted_feature_dict[idx], logits_dict[idx], labels_dict[idx] = local.train(net = copy.deepcopy(net_glob_client).to(device))
        # w_locals_client.append(copy.deepcopy(w_client))
        end =  time.perf_counter() 
        time_client[idx].append(end-begin)
        # print(time_client)
        # Testing -------------------
        extracted_feature_dict_test[idx], labels_dict_test[idx] = local.evaluate(net = copy.deepcopy(net_glob_client).to(device), ell= iter)
        # ------
        server_trainer.add_local_trained_result(idx, extracted_feature_dict[idx], logits_dict[idx], labels_dict[idx],
                                                 extracted_feature_dict_test[idx], labels_dict_test[idx])
        # b_all_received = server_trainer.check_whether_all_receive()
        # logging.info("b_all_received = " + str(b_all_received))

    begin =  time.perf_counter() 
    server_trainer.train()
    end =  time.perf_counter() 
    time_server.append(end-begin)
    for idx in idxs_users:
        server_logits_dict[idx] = server_trainer.get_global_logits(idx)

    # Evaluate for one epoch on validation set
    test_metrics = server_trainer.eval_large_model_on_the_server()

    # Find the best accTop1 model.
    test_acc = test_metrics['test_accTop1']
    # last_path = os.path.join('./checkpoint/last.pth')
    # Save latest model weights, optimizer and accuracy
    # torch.save({'state_dict': self.model_global.state_dict(),
    #             'optim_dict': self.optimizer.state_dict(),
    #             # 'epoch': round_idx + 1,
    #             'test_accTop1': test_metrics['test_accTop1'],
    #             'test_accTop5': test_metrics['test_accTop5']}, last_path)
    print("-----------------------------------------------------------")
    # print("------ FedServer: Federation process at Client-Side ------- ")
    # print("-----------------------------------------------------------")
    print("Test/Loss", test_metrics['test_loss'])
    print('test_accTop1', test_metrics['test_accTop1'])
    print('test_accTop5', test_metrics['test_accTop5'])

    # Update client-side global model 
    # net_glob_client.load_state_dict(w_glob_client)    
    
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
print('finally:',avg_time_client,std_time_client)

round_process = [i for i in range(1, len(acc_train_collect)+1)]
df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect, 
                'avg_time_client':avg_time_client, 'std_time_client': std_time_client, 'time_server':sum(time_server) })     
file_name = program+".xlsx"    
df.to_excel(file_name, sheet_name= "v1_test", index = False)    


#=============================================================================
#                         Program Completed
#=============================================================================