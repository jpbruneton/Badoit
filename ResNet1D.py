#  ================ AlphaZero algorithm for Connect 4 game =================== #
# Name:             ResNet.py
# Description:      Includes both a dense and a resnet.
#                   Almost identical/taken from https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html
# Authors:          Jean-Philippe Bruneton & Adèle Douin & Vincent Reverdy
# Date:             2018
# License:          BSD 3-Clause License
# ============================================================================ #

# ================================= PREAMBLE ================================= #
# Packages
import torch
import torch.utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data
import time
import math
from collections import OrderedDict
import torch.utils.data
import numpy as np
import config
import random

# ================================= CLASS : basic ResNet Block ================================= #


# ================================= CLASS : ResNet + two heads ================================= #

#no bias in conv
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

#dc inplanes = convsize
#batchnorm 1D

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        m = OrderedDict()
        m['conv1'] = conv3x3(inplanes, planes, stride)
        m['bn1'] = nn.BatchNorm1d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(planes, planes)
        m['bn2'] = nn.BatchNorm1d(planes)
        self.group1 = nn.Sequential(m)

        self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual
        out = self.relu(out)

        return out

#=======================================================#

class ResNet(nn.Module):
    def __init__(self, ins, outs, block, layers):

        self.input_dim = ins
        self.output_dim = outs

        self.inplanes = config.convsize
        self.convsize = config.convsize
        super(ResNet, self).__init__()

        torch.set_num_threads(1)

        #as a start : the three features are mapped into a conv with 4*4 kernel
        self.ksize = 3
        self.padding = 1
        m = OrderedDict()

        m['conv1'] = nn.Conv1d(1, self.convsize, kernel_size = self.ksize, stride=1, padding=self.padding, bias=False)
        m['bn1'] = nn.BatchNorm1d(self.convsize)
        m['relu1'] = nn.ReLU(inplace=True)

        self.group1= nn.Sequential(m)

        #next : entering the resnet tower
        self.layer1 = self._make_layer(block, self.convsize, layers)

        #next : entering the policy head
        pol_filters = config.polfilters
        self.policy_entrance = nn.Conv1d(self.convsize, config.polfilters, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnpolicy = nn.BatchNorm1d(config.polfilters)
        self.relu_pol = nn.ReLU(inplace=True)


        #if dense layer in policy head
        if config.usehiddenpol:
            self.hidden_dense_pol = nn.Linear(pol_filters * self.input_dim, config.hiddensize)
            self.relu_hidden_pol = nn.ReLU(inplace=True)
            self.fcpol1 = nn.Linear(config.hiddensize, self.output_dim)
        else:
            self.fcpol2= nn.Linear(pol_filters*self.input_dim, self.output_dim)

        self.softmaxpol=nn.Softmax(dim=1)
        #end of policy head


        # in parallel: entering the value head
        val_filters = config.valfilters
        self.value_entrance = nn.Conv1d(self.convsize, config.valfilters, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnvalue = nn.BatchNorm1d(config.valfilters)
        self.relu_val = nn.ReLU(inplace=True)

        #entering a dense hidden layer
        self.hidden_dense_value = nn.Linear(val_filters * self.input_dim, config.hiddensize)
        self.relu_hidden_val = nn.ReLU(inplace=True)
        self.fcval =  nn.Linear(config.hiddensize, 1)
        self.qval=nn.Tanh()
        #end value head

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0]* m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / (5*n)))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if type(x) == np.ndarray :
            x = torch.FloatTensor(x)
            x = torch.unsqueeze(torch.unsqueeze(x, 0), 0)
            #x = torch.unsqueeze(x,0)

        x = self.group1(x)
        x = self.layer1(x)
        #print('aftertower', x.shape)

        x1 = self.policy_entrance(x)
        #print('afterpolentrance', x1.shape)

        x1 = self.bnpolicy(x1)
        x1 = self.relu_pol(x1)
        x1 = x1.view(-1, config.polfilters*self.input_dim)
        #print('afterflattenpol', x1.shape)


        if config.usehiddenpol:
            x1 = self.hidden_dense_pol(x1)
            x1 = self.relu_hidden_pol(x1)
            x1 = self.fcpol1(x1)
        else:
            x1 = self.fcpol2(x1)

        x1 = self.softmaxpol(x1)

        x2 = self.value_entrance(x)
        x2 = self.bnvalue(x2)
        x2 = self.relu_val(x2)
        x2 = x2.view(-1, self.input_dim*config.valfilters)
        x2 = self.hidden_dense_value(x2)
        x2 = self.relu_hidden_val(x2)
        x2 = self.fcval(x2)
        x2 = self.qval(x2)

        return x2, x1

# -----------------------------------------------------------------#
# builds the model
def resnet18(ins, outs, pretrained=False, model_root=None, **kwargs):
    model = ResNet(ins, outs, BasicBlock, config.res_tower, **kwargs)
    return model



# ================================= CLASS : ResNet training ================================= #

class ResNet_Training:
    # -----------------------------------------------------------------#
    def __init__(self, net, batch_size, n_epoch, learning_rate, train_set, test_set, num_worker):
        self.net = net
        self.batch_size = batch_size
        self.n_epochs = n_epoch
        self.learning_rate = learning_rate
        self.num_worker = num_worker
        torch.set_num_threads(1)

        if config.use_cuda:
            self.net = self.net.cuda()

        self.train_set = train_set
        self.test_set = test_set

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True
                                                        , num_workers=self.num_worker, drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=64, shuffle=True
                                                       , num_workers=self.num_worker)
        self.valid_loader = torch.utils.data.DataLoader(self.train_set, batch_size=128, shuffle=True
                                                        , num_workers=self.num_worker)
        self.net.train()

    # -----------------------------------------------------------------#
    # Losses
    def Loss_value(self):
        loss = torch.nn.MSELoss()
        return loss

    def Loss_policy_bce(self):
        loss = torch.nn.BCELoss()
        return loss

    # -----------------------------------------------------------------#
    # Optimizers
    def Optimizer(self):

        if config.optim == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=config.momentum,
                                  weight_decay=config.wdecay)
        elif config.optim == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=config.wdecay)

        elif config.optim == 'rms':
            optimizer = optim.RMSprop(self.net.parameters(), lr=self.learning_rate, momentum=config.momentum,
                                      weight_decay=config.wdecay)

        return optimizer

    # -----------------------------------------------------------------#
    # training function

    def trainNet(self):

        n_batches = len(self.train_loader)
        print(n_batches, 'batches')
        optimizer = self.Optimizer()

        # Loop for n_epochs
        for epoch in range(self.n_epochs):

            running_loss = 0.0
            print_every = n_batches // 2
            start_time = time.time()
            total_train_loss = 0

            for i, data in enumerate(self.train_loader, 0):

                inputs = data[:, :, 0:self.net.input_dim]
                probas = data[:, :, self.net.input_dim:self.net.input_dim + self.net.output_dim:1]
                reward = data[:, :, -1]


                probas = probas.float()
                reward = reward.float()
                reward = reward.view(self.batch_size, 1)

                if config.use_cuda:
                    inputs, probas, reward = inputs.cuda(), probas.cuda(), reward.cuda()

                inputs = Variable(inputs.float())
                probas, reward = Variable(probas), Variable(reward)

                # Set the parameter gradients to zero
                optimizer.zero_grad()

                # Forward pass, backward pass, optimize
                vh, ph = self.net(inputs)
                loss = 0
                loss += self.Loss_value()(vh, reward)
                loss += self.Loss_policy_bce()(ph, probas)

                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.data.item()
                total_train_loss += loss.data.item()

                if (i + 1) % (print_every + 1) == 0:
                    print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))

        #    #Reset running loss and time
                    running_loss = 0.0
                    start_time = time.time()

        # send back the model to cpu for next self play games using forward in parallel cpu
        self.net.cpu()
