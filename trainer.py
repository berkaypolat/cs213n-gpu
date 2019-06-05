from utils import *
import torch.optim as optim
import torch
import numpy as np
from tqdm import tqdm

class CheXpertTrainer():
    
    def __init__(self, model, class_names, use_cuda, device, epoch = 1, checkpoint = None):
        self.model = model
        self.class_names = class_names
        self.epoch = epoch
        self.checkpoint = checkpoint
        self.use_cuda = use_cuda
        self.device = device

    def train (self, dataLoaders):
        
        #SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
                
        #SETTINGS: LOSS
        loss = torch.nn.BCELoss(reduction = 'mean')
        
        #LOAD CHECKPOINT 
        if self.checkpoint != None:
            load_checkpoint(self.checkpoint, self.model, optimizer, self.use_cuda)

        budget = 0.3  ##we need work on hyperparameter search for this!
        
        #TRAIN THE NETWORK
        
        loss_train = []
        loss_eval = []
        
        
        for epochID in range(0, self.epoch):
            
            
            loss_tr, lmbda = self._epochTrain(loss, budget,optimizer, dataLoaders[0], epochID)
            loss_ev = self._epochEval(loss, lmbda, dataLoaders[1], epochID)
            
            loss_train.extend(loss_tr)
            loss_eval.extend(loss_ev)
            
            print('Train Loss after epoch ', epochID + 1, ': ', loss_tr[-1])
            print('Eval Loss after epoch ', epochID + 1, ': ', loss_ev[-1])
            
            #compute average auroc scores for both train and validation sets
            #(only computing validation because train is too long)
            print('AUROC scores after epoch ', epochID + 1, ':')
#             print('Training set: ')
#             _,_ = self.test(dataLoaders[0])
            print('Validation set: ')
            labels,predictions,aurocIndividual = self.test(dataLoaders[1])
            labels = labels.cpu().numpy()
            predictions = predictions.cpu().numpy()
            
            torch.save({'state_dict':self.model.state_dict(), 'optimizer':optimizer.state_dict(),
                        'train_losses':loss_train, 'eval_losses':loss_eval, 'aurocResults':aurocIndividual,
                        'labels':labels, 'predictions':predictions},
                        'checkpoints/unpretrained_epoch' + str(epochID + 1) + '.pth.tar')
            
         
        return loss_train, loss_eval    
    #-------------------------------------------------------------------------------- 
       
    def _epochTrain(self, loss, budget, optimizer, dataLoaderTrain, epochID):
        
        self.model.train()
        losstrain = []
        
        lmbda = 0.1    #start with reasonable value

        progress_bar = tqdm(dataLoaderTrain)
        
        for batchID, (varInput, target) in enumerate(progress_bar):
            progress_bar.set_description("Train Epoch #" + str(epochID+1))
            
            #batch.append(batchID)
            varTarget = torch.stack(target).float().transpose(0,1).to(self.device)

            #varTarget = target.cuda(non_blocking = True)        

            bs, c, h, w = varInput.size()
            varInput = varInput.view(-1, c, h, w)

            optimizer.zero_grad()
            
            varOutput, confidence = self.model(varInput)
            confidence = torch.sigmoid(confidence)
            #print('tensor: ', confidence)
            
            # prevent any numerical instability
            eps = 1e-12
            varOutput = torch.clamp(varOutput, 0. + eps, 1. - eps)
            confidence = torch.clamp(confidence, 0. + eps, 1. - eps)
            #print('tensor: ', confidence)
            
            # Randomly set half of the confidences to 1 (i.e. no hints)
            b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).to(self.device)
            conf = confidence * b + (1 - b)
            pred_new = varOutput * conf + varTarget * (1 - conf)
            
            first_loss = loss(pred_new, varTarget)
            
            #maybe this confidence loss makes more sense
# #             print('tensor: ', confidence)
# #             confidence = varTarget * confidence
# #             print('Original tensor: ', confidence)
# #             print('Row-wise sum: ', torch.sum(confidence,1))
# #             print('# of non-zeros: ', torch.sum(confidence != 0 ,1))
                
#             confidence = torch.sum(confidence,1) / torch.sum(confidence != 0, 1).float()
#             #confidence = torch.mean(confidence,1)
#             print('New confidence tensor: ', confidence)
            
#             confidence = torch.clamp(confidence, 0. + eps, 1. - eps)
#             confidence = -torch.log(confidence)

#             second_loss = torch.mean(confidence)
            second_loss = torch.mean(-torch.log(confidence))
            loss_value = first_loss + lmbda * second_loss
#             print("First Loss: ", first_loss)
#             print("Second Loss: ",second_loss)
#             print("Total Loss: ",loss_value)
#             print("===================")
            
            if budget > second_loss.item():
                lmbda = lmbda / 1.01
            elif budget <= second_loss.item():
                lmbda = lmbda / 0.99
            
            
            
            loss_value.backward()
            optimizer.step()
            
            l = loss_value.item()
            losstrain.append(l)
            
        return losstrain, lmbda
    
    #-------------------------------------
    
    def _epochEval(self, loss, lmbda, dataLoaderEval, epochID):
    
        self.model.eval()
        
        lossVal = []
        
        with torch.no_grad():
            progress_bar = tqdm(dataLoaderEval)
            for batchID, (varInput, target) in enumerate(progress_bar):
                progress_bar.set_description("Eval Epoch #" + str(epochID+1))
            
                varTarget = torch.stack(target).float().transpose(0,1).to(self.device)

                #varTarget = target.cuda(non_blocking = True    

                bs, c, h, w = varInput.size()
                varInput = varInput.view(-1, c, h, w)

                varOutput, confidence = self.model(varInput)
                confidence = torch.sigmoid(confidence)


                # prevent any numerical instability
                eps = 1e-12
                varOutput = torch.clamp(varOutput, 0. + eps, 1. - eps)
                confidence = torch.clamp(confidence, 0. + eps, 1. - eps)

                # Randomly set half of the confidences to 1 (i.e. no hints)
                b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).to(self.device)
                conf = confidence * b + (1 - b)
                pred_new = varOutput * conf + varTarget * (1 - conf)

                first_loss = loss(pred_new, varTarget)
#                 confidence = varTarget * confidence
#                 confidence = torch.mean(confidence,1)
#                 confidence = torch.clamp(confidence, 0. + eps, 1. - eps)
#                 confidence = -torch.log(confidence)
#                 second_loss = torch.mean(confidence)
                
                second_loss = torch.mean(-torch.log(confidence))
                loss_value = first_loss + lmbda * second_loss
                
                lossVal.append(loss_value.item())
                
        return lossVal
    
    #-------------------------------------------------
    
    def test(self, dataLoaderSet):
        if self.use_cuda:
                outGT = torch.FloatTensor().to(self.device)
                outPRED = torch.FloatTensor().to(self.device)
        else:
            outGT = torch.FloatTensor()
            outPRED = torch.FloatTensor()

        self.model.eval()

        with torch.no_grad():
            progress_bar = tqdm(dataLoaderSet)
            for i, (data, target) in enumerate(progress_bar):
                progress_bar.set_description("Test Progress..")
                target = torch.stack(target).float().transpose(0,1).to(self.device)
                outGT = torch.cat((outGT, target), 0).to(self.device)

                bs, c, h, w = data.size()
                varInput = data.view(-1, c, h, w)

                out,_ = self.model(varInput)
                outPRED = torch.cat((outPRED, out), 0)
                
        aurocIndividual = computeAUROC(outGT, outPRED, len(self.class_names))
        aurocMean = np.array(aurocIndividual).mean()

        print ('AUROC mean ', aurocMean)

        for i in range (0, len(aurocIndividual)):
            print (self.class_names[i], ' ', aurocIndividual[i])

        return outGT, outPRED, aurocIndividual
    
    
    
    
    
        