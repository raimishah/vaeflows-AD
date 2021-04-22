import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import argparse
import dill

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
from tcn_trainer import TCN_Trainer

import utils

from utils import softclip
from models.cnn_sigmaVAE_msl import CNN_sigmaVAE
from models.cnn_sigmacVAE_msl import CNN_sigmacVAE
from models.tcn_vae import TCN_VAE 


import evaluation_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_scores_and_labels(model_type, model, df_Y_test, dataloader, X_test_tensor):
    
    if 'tcn' in model_type:
        preds, scores, mse = evaluation_utils.evaluate_model_tcn(model, model_type, dataloader, X_test_tensor)
    else:
        preds, scores, mse = evaluation_utils.evaluate_model_new(model, model_type, dataloader, X_test_tensor)
    
    labels = df_Y_test.values
    labels = np.reshape(labels, (labels.shape[0], ))
    anomaly_idxs = np.where(labels == 1)[0]
    
    #create labels only up to num preds
    labels = np.zeros(len(scores), dtype=np.int)
    labels[anomaly_idxs] = 1

    return scores, labels, mse

def train_and_eval_on_SMAP(model_type, model, num_epochs, learning_rate, window_size, cond_window_size, batch_size, early_stop_patience=100, use_validation=False):

    machine_name = 'SMAP'
    valloader=None
    if 'cvae' not in model_type:
        if not use_validation:
            X_train_data, X_test_data, X_train_tensor, X_test_tensor, df_Y_test, trainloader, testloader = utils.read_machine_data('../../datasets/ServerMachineDataset/' + machine_name, window_size, model.jump_size, batch_size)
        else:
            X_train_data, X_test_data, X_train_tensor, X_test_tensor, df_Y_test, trainloader, valloader, testloader = utils.read_machine_data_with_validation('../../datasets/ServerMachineDataset/' + machine_name, window_size, model.jump_size, batch_size, val_size=.3)


    else:
        if not use_validation:
            X_train_data, X_test_data, X_train_tensor, cond_train_tensor, X_test_tensor, cond_test_tensor, df_Y_test, trainloader, testloader = utils.read_machine_data_cvae('../../datasets/ServerMachineDataset/' + machine_name, window_size, cond_window_size, model.jump_size, batch_size)
        else:
            X_train_data, X_test_data, X_train_tensor, cond_train_tensor, X_test_tensor, cond_test_tensor, df_Y_test, trainloader, valloader, testloader = utils.read_machine_data_cvae_with_validation('../../datasets/ServerMachineDataset/' + machine_name, window_size, cond_window_size, model.jump_size, batch_size, val_size=.3)

    print(X_train_tensor.shape)

    if model_type=='tcn_vae' or model_type=='tcn_cvae':
        trainer = TCN_Trainer(data_name = 'msl', model_type = model_type, flow_type=model.flow_type, early_stop_patience=early_stop_patience)
    elif model_type=='vae' or model_type=='cvae':
        trainer = Trainer(data_name = 'msl', model_type = model_type, flow_type=model.flow_type, early_stop_patience=early_stop_patience)
    
    print('Starting training......\n')
    model, flag = trainer.train_model(model, num_epochs=num_epochs, learning_rate=learning_rate, trainloader=trainloader, valloader=valloader)
    if flag:
        print('Failed to train -- returning')
        return

    if model.prob_decoder:
        save_folder = 'saved_models/SMAP/' + model_type + model.flow_type + '_prob_decoder/' if model.flow_type != None else 'saved_models/SMAP/' + model_type + '_prob_decoder/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        torch.save(model, save_folder + model_type + '-' + machine_name + '.pth')
    else:
        save_folder = 'saved_models/SMAP/' + model_type + model.flow_type + '/' if model.flow_type != None else 'saved_models/SMAP/' + model_type + '/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        torch.save(model, save_folder + model_type + '-' + machine_name + '.pth', pickle_module=dill)

    print('saving to folder {}'.format(save_folder))

    #plot loss also
    trainer.plot_model_loss(save_folder, machine_name)

    #utils.plot_reconstruction(model, model_type='vae',dataloader=trainloader)
    #utils.plot_reconstruction(model, model_type='vae',dataloader=testloader)

    #EVALUATION
    #if model_type=='vae':
    #    scores, labels, mse = get_scores_and_labels(model_type, model, df_Y_test, testloader, X_test_tensor)

    #if model_type=='cvae':
    #    scores, labels, mse = get_scores_and_labels(model_type, model, df_Y_test, testloader, X_test_tensor)

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

    print('\n\n EVT AD:')
    confusion_matrix_metrics, alert_delays = evaluation_utils.compute_metrics_with_pareto(labels, scores, .01)
    tn_fp_fn_tp=np.empty((0,4)) 
    tn_fp_fn_tp = np.concatenate([tn_fp_fn_tp, confusion_matrix_metrics])
    
    tn = tn_fp_fn_tp[:, 0].sum()
    fp = tn_fp_fn_tp[:, 1].sum()
    fn = tn_fp_fn_tp[:, 2].sum()
    tp = tn_fp_fn_tp[:, 3].sum()
    
    F1 = tp / (tp+.5*(fp+fn))
    print('EVT AD best : {}'.format(F1)) 

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

    model_type='tcn_cvae' #'vae', 'cvae', 'tcn_vae'
    conditional=False
    if 'cvae' in model_type:
        conditional=True
    flow_type=None
    prob_decoder=False
    batch_size=256
    jump_size=32
    latent_dim=10
    num_feats=25
    window_size=32 if model_type=='tcn_vae' else 100
    num_epochs=1
    lr = .005 if flow_type==None else .005
    early_stop_patience=300 if flow_type==None else 500
    kernel_size=5
    num_levels=2
    convs_per_level=2


    if jump_size > window_size:
        print('jump_size cannot be greater than window size.')
        return
    if kernel_size%2==0:
        print('kernel_size must be odd. (try 3,5,7)')
        return
    if flow_type=='DSF' and 'cvae' not in model_type:
        print("Need model_type=='cvae' for using DSF.")
        return

    print('Training with {}, with flow - {}, and prob decoder - {}'.format(model_type, flow_type, prob_decoder))



    if model_type=='vae':
        cond_window_size=-1
        model = CNN_sigmaVAE(latent_dim=latent_dim, window_size=window_size, jump_size=jump_size, num_feats=num_feats, flow_type=flow_type, use_probabilistic_decoder=prob_decoder).to(device)

    elif model_type=='cvae':	
        cond_window_size=13
        model = CNN_sigmacVAE(latent_dim=latent_dim, window_size=window_size, cond_window_size=cond_window_size, jump_size=jump_size, num_feats=num_feats, flow_type=flow_type, use_probabilistic_decoder=prob_decoder).to(device)
            
    elif model_type=='tcn_vae':
        cond_window_size=-1
        model = TCN_VAE(conditional=conditional, latent_dim=latent_dim, window_size=window_size, cond_window_size=cond_window_size, jump_size=jump_size, num_feats=num_feats, kernel_size=kernel_size, num_levels=num_levels, convs_per_level=convs_per_level, flow_type=flow_type, use_probabilistic_decoder=prob_decoder).to(device)

    elif model_type=='tcn_cvae':
        cond_window_size=num_feats
        model = TCN_VAE(conditional=conditional, latent_dim=latent_dim, window_size=window_size, cond_window_size=cond_window_size, jump_size=jump_size, num_feats=num_feats, kernel_size=kernel_size, num_levels=num_levels, convs_per_level=convs_per_level, flow_type=flow_type, use_probabilistic_decoder=prob_decoder).to(device)

    print(model)

    train_and_eval_on_SMAP(model_type, model, num_epochs, lr, window_size, cond_window_size, batch_size, early_stop_patience=early_stop_patience, use_validation=True)

if __name__=='__main__':
	main()