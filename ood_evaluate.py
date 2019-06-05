import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def evaluate_ood(data_loader, mode, model, device):
    model.eval()
    outPred = []
    outGround = []
    outConf = []
    nih = False
    bxent = nn.BCELoss(reduction = 'mean')
    
    progress_bar = tqdm(data_loader)
    
    for i, (images,labels) in enumerate(progress_bar):

        if type(labels) == list:
            labels = torch.stack(labels).float().transpose(0,1).to(device)
        else:
            nih = True


        bs, c, h, w = images.size()
        varInput = images.view(-1, c, h, w)
        
        outGround.append(labels.cpu().numpy())

        if mode == 'confidence':
            #with torch.no_grad():
            
            T = 1000  #this hyperparameter can also be experimeted
            epsilon = 0.001

            model.zero_grad()
            varInput.requires_grad_()
            
            pred, _ = model(varInput)
           
            if nih:
                labels = torch.zeros(pred.shape[0], pred.shape[1]).to(device)
           
            pred = pred / T
            
            loss = bxent(pred, labels)
            loss.backward()

            varInput = varInput - epsilon * torch.sign(varInput.grad)
            #pred, _ = model(varInput)
            preds, confidence = model(varInput)
            confidence = torch.sigmoid(confidence)
            confidence = confidence.data.cpu().numpy()
            outPred.append(preds.data.cpu().numpy())
            outConf.append(confidence)

        elif mode == 'confidence_scaling':
            epsilon = 0.001  ##value needs to be determined (noise magnitude) 

            model.zero_grad()
            varInput.requires_grad_()
            
            _,confidence = model(varInput)
            confidence = torch.sigmoid(confidence)
            
            loss = torch.mean(-torch.log(confidence))
            loss.backward()

            varInput = varInput - epsilon * torch.sign(varInput.grad)

            preds,confidence = model(varInput)
            confidence = torch.sigmoid(confidence)
            confidence = confidence.data.cpu().numpy()
            outPred.append(preds.data.cpu().numpy())
            outConf.append(confidence)


        elif mode == 'baseline':
            #uses the non-augmented model architecture
            with torch.no_grad():
                pred, _ = model(varInput)
                pred = pred.cpu().numpy()
                outPred.append(pred)

        elif mode == 'odin':
            #uses the non-augmented model architecture same as baseline
            T = 1000  #this hyperparameter can also be experimeted
            epsilon = 0.001

            model.zero_grad()
            varInput.requires_grad_()
            
            pred, _ = model(varInput)
           
            if nih:
                labels = torch.zeros(pred.shape[0], pred.shape[1]).to(device)
           
            pred = pred / T
            
            loss = bxent(pred, labels)
            loss.backward()

            varInput = varInput - epsilon * torch.sign(varInput.grad)
            pred, _ = model(varInput)

            #might need to take Sigmoid layer out from the model class for ODIN

            pred = pred.data.cpu().numpy()
            outPred.append(pred)
            
            
    outPred = np.concatenate(outPred)
    outGround = np.concatenate(outGround)
    outConf = None if mode == "odin" or mode == "baseline" else np.concatenate(outConf)
    return outPred, outGround, outConf