#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import torch.nn.functional as func
from util_datasets import GaussianNoise, UniformNoise
import csv

from sklearn.metrics.ranking import roc_auc_score
import sklearn.metrics as metrics
import random


use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")


class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

imgtransResize = (320, 320)
imgtransCrop = 224


# In[2]:


from trainer import CheXpertTrainer 
from chexpertClass import CheXpertData
from denseNet121 import DenseNet121
from utils import load_checkpoint


# In[3]:


#TRANSFORM DATA SEQUENCE
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
#transformList.append(transforms.Resize(imgtransCrop))
transformList.append(transforms.RandomResizedCrop(imgtransCrop))
transformList.append(transforms.RandomHorizontalFlip())
transformList.append(transforms.ToTensor())
transformList.append(normalize)      
transformSequence=transforms.Compose(transformList)


# In[4]:


#CheXpert dataset loadig
chex_datasetValid = CheXpertData('datasets/chexpert-small/CheXpert-v1.0-small/valid.csv' ,transformSequence, preload = True, policy="ones")
chex_datasetTrain = CheXpertData('datasets/chexpert-small/CheXpert-v1.0-small/train.csv' ,transformSequence, policy="ones")
datasetTest, datasetTrain = random_split(chex_datasetTrain, [500, len(chex_datasetTrain) - 500])
#for model train testing purposes
chex_valid = torch.utils.data.ConcatDataset([chex_datasetValid, datasetTest])
dataLoaderChex = DataLoader(dataset=chex_valid, batch_size=10, shuffle=True,  num_workers=0, pin_memory=True)


# In[5]:


#NIH dataset loading
nih_dataset = datasets.ImageFolder(root='datasets/nih-small/small', transform = transformSequence)
nih_test, nih_train = random_split(nih_dataset, [734, len(nih_dataset) - 734])
dataLoaderNIH = DataLoader(dataset=nih_test, batch_size=64, shuffle=False,  num_workers=2, pin_memory=True)


# In[6]:


model = DenseNet121(len(class_names)).to(device)


# In[7]:


model = torch.nn.DataParallel(model).to(device)


# In[8]:


checkpoint_path = 'cheXpert_github/model_ones_3epoch_densenet.tar'
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
load_checkpoint(checkpoint_path, model, optimizer, use_cuda)


# In[9]:


from utils import load_checkpoint
import torch.optim as optim
import torch

class CheXpertTrainer():

    def train (self, model, dataLoaderTrain, nnClassCount, trMaxEpoch, checkpoint, use_cuda):
        
        #SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
                
        #SETTINGS: LOSS
        loss = torch.nn.BCELoss(size_average = True)
        
        #LOAD CHECKPOINT 
        if checkpoint != None:
            load_checkpoint(checkpoint, model, optimizer, use_cuda)

        budget = 0.3
        
        #TRAIN THE NETWORK
        
        for epochID in range(0, trMaxEpoch):
            
            
            batchs, losst = CheXpertTrainer.epochTrain(model, dataLoaderTrain, optimizer, trMaxEpoch, nnClassCount, loss, budget)
         
        return batchs, losst #, losse        
    #-------------------------------------------------------------------------------- 
       
    def epochTrain(model, dataLoader, optimizer, epochMax, classCount, loss, budget):
        
        batch = []
        losstrain = []
        losseval = []
        
        lmbda = 0.1    #start with reasonable value
        
        model.train()

        for batchID, (varInput, target) in enumerate(dataLoader):
            
            batch.append(batchID)
            varTarget = torch.stack(target).float().transpose(0,1).to(device)
            print(varTarget.shape)
            #varTarget = target.cuda(non_blocking = True)
            
            #varTarget = target.cuda()         

            bs, c, h, w = varInput.size()
            varInput = varInput.view(-1, c, h, w)

            varOutput, confidence = model(varInput)
            confidence = torch.sigmoid(confidence)
            
            
            # prevent any numerical instability
            eps = 1e-12
            varOutput = torch.clamp(varOutput, 0. + eps, 1. - eps)
            confidence = torch.clamp(confidence, 0. + eps, 1. - eps)
            
            # Randomly set half of the confidences to 1 (i.e. no hints)
            b = Variable(torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1))).to(device)
            conf = confidence * b + (1 - b)
            pred_new = varOutput * conf + varTarget * (1 - conf)
            
            first_loss = loss(pred_new, varTarget)
            second_loss = torch.mean(torch.mean(-torch.log(confidence),1))
            
            loss_value = first_loss + lmbda * second_loss
            
            if budget > second_loss.item():
                lmbda = lmbda / 1.01
            elif budget <= second_loss.item():
                lmbda = lmbda / 0.99
            
            
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            l = loss_value.item()
            losstrain.append(l)
            print(l)
            
        return batch, losstrain #, losseval


# In[ ]:


trainer = CheXpertTrainer()
epochs = 1
batchs, lost_train = trainer.train(model,dataLoaderChex, len(class_names), epochs, checkpoint_path, use_cuda)


# In[ ]:




