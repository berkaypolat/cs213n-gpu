import torch
import numpy as np
from sklearn.metrics.ranking import roc_auc_score


#this is private function used only for loading the parameters from pretrained DenseNet121 to
#new model with the confidence branch
def load_checkpoint(checkpoint_path, model, optimizer, use_cuda):
    if use_cuda:
        state = torch.load(checkpoint_path)
    else:
        state = torch.load(checkpoint_path, map_location='cpu')
    
    state_dict = state['state_dict']
    params_list = state['optimizer']['param_groups'][0]['params']
    
    model_state = model.state_dict()
    model_state.update(state_dict)
    model.load_state_dict(model_state)
    
    optim_dict = optimizer.state_dict()
    param1 = optim_dict['param_groups'][0]['params'][-1]
    param2 = optim_dict['param_groups'][0]['params'][-2]
    new_param_lst = params_list.copy()
    new_param_lst.append(param2)
    new_param_lst.append(param1)
    optim_dict['state'] = state['optimizer']['state']
    optim_dict['param_groups'][0]['params'] = new_param_lst
    optimizer.load_state_dict(optim_dict)
    
    print('model loaded from %s' % checkpoint_path)
    
#use this loader to load any of the epoch(-).ph.tar files from the checkpoints folder 
#for the densenet121 model with confidence branch
def load_checkpoint_v2(checkpoint_path, model, optimizer, use_cuda):
    if use_cuda:
        state = torch.load(checkpoint_path)
    else:
        state = torch.load(checkpoint_path, map_location='cpu')
   
    model_state = state['state_dict']
    optim_state = state['optimizer']
    train_loss = state['train_losses']
    eval_loss = state['eval_losses']
    auroc_scores = state['aurocResults']
    labels = state['labels']
    scores = state['predictions']
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optim_state)
    
    print('model loaded from %s' % checkpoint_path)
    
    return train_loss, eval_loss, labels, scores
    
def computeAUROC (labels, predictions, class_count):
        
        outAUROC = []
        
        np_labels = labels.cpu().numpy()
        np_preds = predictions.cpu().numpy()
        
        for i in range(class_count):
            try:
                outAUROC.append(roc_auc_score(np_labels[:, i], np_preds[:, i]))
            except ValueError:
                pass
        return outAUROC
    
def tpr76(ind_confidences, ood_confidences):
    #calculate the falsepositive error when tpr is 75-76%
    Y1 = ood_confidences
    X1 = ind_confidences

    start = np.min([np.min(X1), np.min(Y1)])
    end = np.max([np.max(X1), np.max(Y1)])
    gap = (end - start) / 100000

    total = 1.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        #if tpr <= 0.9505 and tpr >= 0.9495:
        if tpr <= 0.7605 and tpr >= 0.7495:
            fpr += error2
            total += 1

    fprBase = fpr / total

    return fprBase


def detection(ind_confidences, ood_confidences, n_iter=100000, return_data=False):
    # calculate the minimum detection error
    Y1 = ood_confidences
    X1 = ind_confidences

    start = np.min([np.min(X1), np.min(Y1)])
    end = np.max([np.max(X1), np.max(Y1)])
    gap = (end - start) / n_iter

    best_error = 1.0
    best_delta = None
    all_thresholds = []
    all_errors = []
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        detection_error = (tpr + error2) / 2.0

        if return_data:
            all_thresholds.append(delta)
            all_errors.append(detection_error)

        if detection_error < best_error:
            best_error = np.minimum(best_error, detection_error)
            best_delta = delta

    if return_data:
        return best_error, best_delta, all_errors, all_thresholds
    else:
        return best_error, best_delta