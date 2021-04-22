import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
from tqdm.notebook import tqdm
from torchsummary import summary
from torchvision import datasets
from torchvision import transforms


from trainer import Trainer

import utils

from utils import softclip
from models.cnn_sigmaVAE_swat import CNN_sigmaVAE
from models.cnn_sigmacVAE import CNN_sigmacVAE


import evaluation_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_scores_and_labels(model_type, model, df_Y_test, dataloader, X_test_tensor):
    
    preds, scores, mse = evaluation_utils.evaluate_model_new(model, model_type, dataloader, X_test_tensor)
    
    labels = df_Y_test.values
    labels = np.reshape(labels, (labels.shape[0], ))
    anomaly_idxs = np.where(labels == 1)[0]
    
    #create labels only up to num preds
    labels = np.zeros(len(scores), dtype=np.int)
    labels[anomaly_idxs] = 1

    return scores, labels, mse

def train_and_eval_on_SMAP(model_type, model, num_epochs, learning_rate, window_size, cond_window_size, batch_size, early_stop_patience=100, use_validation=False):

    machine_name = 'swat'
    valloader=None
    if 'cvae' not in model_type:
        if not use_validation:
            X_train_data, X_test_data, X_train_tensor, X_test_tensor, df_Y_test, trainloader, testloader = utils.read_machine_data('../../datasets/SWaT/' + machine_name, window_size, batch_size)
        else:
            X_train_data, X_test_data, X_train_tensor, X_test_tensor, df_Y_test, trainloader, valloader, testloader = utils.read_machine_data_with_validation('../../datasets/SWaT/' + machine_name, window_size, batch_size, val_size=.3)


    else:
        if not use_validation:
            X_train_data, X_test_data, X_train_tensor, cond_train_tensor, X_test_tensor, cond_test_tensor, df_Y_test, trainloader, testloader = utils.read_machine_data_cvae('../../datasets/SWaT/' + machine_name, window_size, cond_window_size, batch_size)
        else:
            X_train_data, X_test_data, X_train_tensor, cond_train_tensor, X_test_tensor, cond_test_tensor, df_Y_test, trainloader, valloader, testloader = utils.read_machine_data_cvae_with_validation('../../datasets/SWaT/' + machine_name, window_size, cond_window_size, batch_size, val_size=.3)

    print(X_train_tensor.shape)
    print(X_test_tensor.shape)
    print(df_Y_test.shape)

    trainer = Trainer(data_name = 'smap', model_type = model_type, flow_type=model.flow_type, early_stop_patience=early_stop_patience)
    model, flag = trainer.train_model(model, num_epochs=num_epochs, learning_rate=learning_rate, trainloader=trainloader, valloader=valloader)

    if model.prob_decoder:
        save_folder = 'saved_models/SMAP/' + model_type + model.flow_type + '_prob_decoder/' if model.flow_type != None else 'saved_models/SMAP/' + model_type + '_prob_decoder/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        torch.save(model, save_folder + model_type + '-' + machine_name + '.pth')
    else:
        save_folder = 'saved_models/SMAP/' + model_type + model.flow_type + '/' if model.flow_type != None else 'saved_models/SMAP/' + model_type + '/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        torch.save(model, save_folder + model_type + '-' + machine_name + '.pth')

    print('saving to folder {}'.format(save_folder))

    #plot loss also
    trainer.plot_model_loss(save_folder, machine_name)

    #utils.plot_reconstruction(model, model_type='vae',dataloader=trainloader)
    #utils.plot_reconstruction(model, model_type='vae',dataloader=testloader)


    #evaluation

    if model_type=='vae':
        scores, labels, mse = get_scores_and_labels(model_type, model, df_Y_test, testloader, X_test_tensor)

    if model_type=='cvae':
        scores, labels, mse = get_scores_and_labels(model_type, model, df_Y_test, testloader, X_test_tensor)


    confusion_matrix_metrics, alert_delays = evaluation_utils.compute_AUPR(labels, scores, threshold_jump=500)
    print('[[TN, FP, FN, TP]]')
    print(confusion_matrix_metrics)
    print('Alert Delays : {}'.format(alert_delays))
    print('\n')

    tn_fp_fn_tp=np.empty((0,4)) 
    tn_fp_fn_tp = np.concatenate([tn_fp_fn_tp, confusion_matrix_metrics])
    
    tn = tn_fp_fn_tp[:, 0].sum()
    fp = tn_fp_fn_tp[:, 1].sum()
    fn = tn_fp_fn_tp[:, 2].sum()
    tp = tn_fp_fn_tp[:, 3].sum()
    
    print('TN sum : {}'.format(tn))        
    print('FP sum : {}'.format(fp))       
    print('FN sum : {}'.format(fn))        
    print('TP sum : {}'.format(tp))

    F1 = tp / (tp+.5*(fp+fn))
    print('Overall F1 best : {}'.format(F1)) 


def main():

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type')
    parser.add_argument('--flow_type', nargs='?')
    parser.add_argument('--prob_decoder', default=1)
    args = parser.parse_args()
    model_type = args.model_type
    flow_type = args.flow_type
    prob_decoder = args.prob_decoder

    '''

    model_type='vae'
    flow_type=None
    prob_decoder=True

    print('Training with {}, with flow - {}, and prob decoder - {}'.format(model_type, flow_type, prob_decoder))

    batch_size=256
    latent_dim=10
    num_feats=25
    window_size=100
    num_epochs=10
    lr = .005 if flow_type==None else .0005
    early_stop_patience=100 if flow_type==None else 100

    if model_type=='vae':
        cond_window_size=-1
        model = CNN_sigmaVAE(latent_dim=latent_dim, window_size=window_size, num_feats=num_feats, flow_type=flow_type, use_probabilistic_decoder=prob_decoder).to(device)
        model.cuda() if torch.cuda.is_available() else model.cpu()
        print(model)


    elif model_type=='cvae':	
        cond_window_size=13
        model = CNN_sigmacVAE(latent_dim=latent_dim, window_size=window_size, cond_window_size=cond_window_size ,num_feats=num_feats, flow_type=flow_type, use_probabilistic_decoder=prob_decoder).to(device)
        print(model)
            
    train_and_eval_on_SMAP(model_type, model, num_epochs, lr, window_size, cond_window_size, batch_size, early_stop_patience=early_stop_patience, use_validation=False)

if __name__=='__main__':
	main()