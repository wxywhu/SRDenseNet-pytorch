
# coding: utf-8

# In[7]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np 
import torch.nn.init as init

def xavier(param):
    init.xavier_uniform(param)

class SingleLayer(nn.Module):
    def __init__(self, inChannels,growthRate):
        super(SingleLayer, self).__init__()
        self.conv =nn.Conv2d(inChannels,growthRate,kernel_size=3,padding=1, bias=True)
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class Net(nn.Module):
    def __init__(self,growthRate,nDenselayer):
        super(Net,self).__init__()
        
        self.conv1 = nn.Conv2d(1,growthRate,kernel_size=3, padding=1,bias=True)
        inChannels = growthRate
        
        self.dense1 = self._make_dense(inChannels,growthRate,nDenselayer)
        inChannels += nDenselayer*growthRate

        self.dense2 = self._make_dense(inChannels,growthRate,nDenselayer)
        inChannels += nDenselayer*growthRate
        
        self.dense3 = self._make_dense(inChannels,growthRate,nDenselayer)
        inChannels += nDenselayer*growthRate
        
        self.dense4 = self._make_dense(inChannels,growthRate,nDenselayer)
        inChannels += nDenselayer*growthRate
        
        self.dense5 = self._make_dense(inChannels,growthRate,nDenselayer)
        inChannels += nDenselayer*growthRate
        
        self.dense6 = self._make_dense(inChannels,growthRate,nDenselayer)
        inChannels += nDenselayer*growthRate
        
        self.dense7 = self._make_dense(inChannels,growthRate,nDenselayer)
        inChannels += nDenselayer*growthRate

        self.dense8 = self._make_dense(inChannels,growthRate,nDenselayer)
        inChannels += nDenselayer*growthRate
        
        self.Bottleneck = nn.Conv2d(in_channels=inChannels, out_channels=256, kernel_size=1,padding=0, bias=True)
        
        self.convt1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True)
        
        self.convt2 =nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True)

        self.conv2 =nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3,padding=1, bias=True)
    
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

                        
    def _make_dense(self,inChannels,growthRate, nDenselayer):
        layers = []
        for i in range(int(nDenselayer)):
            layers.append(SingleLayer(inChannels,growthRate))
            inChannels += growthRate
        return nn.Sequential(*layers)
                              
    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = self.dense1(out)
        out = self.dense2(out)
        out = self.dense3(out)
        out = self.dense4(out)
        out = self.dense5(out)
        out = self.dense6(out)
        out = self.dense7(out)
        out = self.dense8(out)
                                         
        out = self.Bottleneck(out)
        out = self.convt1(out)
        out = self.convt2(out)
                                         
        HR = self.conv2(out)
        return HR

