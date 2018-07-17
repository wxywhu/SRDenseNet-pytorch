
# coding: utf-8

# In[16]:


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
    
class SingleBlock(nn.Module):
    def __init__(self, inChannels,growthRate,nDenselayer):
        super(SingleBlock, self).__init__()
        self.block= self._make_dense(inChannels,growthRate, nDenselayer)
        
    def _make_dense(self,inChannels,growthRate, nDenselayer):
        layers = []
        for i in range(int(nDenselayer)):
            layers.append(SingleLayer(inChannels,growthRate))
            inChannels += growthRate
        return nn.Sequential(*layers)
                
    def forward(self, x):
        out=self.block(x)
        return out

class Net(nn.Module):
    def __init__(self,inChannels,growthRate,nDenselayer,nBlock):
        super(Net,self).__init__()
        
        self.conv1 = nn.Conv2d(1,growthRate,kernel_size=3, padding=1,bias=True)
        
        inChannels = growthRate
        
        self.denseblock = self._make_block(inChannels,growthRate, nDenselayer,nBlock)
        inChannels +=growthRate* nDenselayer*nBlock
        
        self.Bottleneck = nn.Conv2d(in_channels=inChannels, out_channels=128, kernel_size=1,padding=0, bias=True)
        
        self.convt1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True)
        
        self.convt2 =nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True)
        #self.convt2 =nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=4, padding=0, bias=True)
        
        self.conv2 =nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3,padding=1, bias=True)
    
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def _make_block(self, inChannels,growthRate, nDenselayer,nBlock):
        blocks =[]
        for i in range(int(nBlock)):
            blocks.append(SingleBlock(inChannels,growthRate,nDenselayer))
            inChannels += growthRate* nDenselayer
        return nn.Sequential(* blocks)  
    
    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = self.denseblock(out)                                      
        out = self.Bottleneck(out)
        out = self.convt1(out)
        out = self.convt2(out)
                                         
        HR = self.conv2(out)
        return HR


# In[17]:

