import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from scipy.stats import norm
from scipy.stats import multivariate_normal


import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchsummary import summary

import torchvision
from torchvision import datasets
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor




def plot_reconstruction(model, model_type, dataloader):
    model.eval()

    dataiter = iter(dataloader)
    x, y = dataiter.next()

    preds = np.empty((0,x.shape[1],x.shape[2],x.shape[3]))
    reals = np.empty((0,x.shape[1],x.shape[2],x.shape[3]))

    cond_window_size=y.shape[2]
    jump_size = model.jump_size

    for j, data in enumerate(dataloader, 0):

        x, y = data
        x = x.cuda() if torch.cuda.is_available() else x.cpu()
        x.to(device)
        y = y.cuda() if torch.cuda.is_available() else y.cpu()
        y.to(device)
        if 'cvae' in model_type:
            outputs, rec_mu, rec_sigma, kl = model(x, y)
        else:
            outputs, rec_mu, rec_sigma, kl = model(x, None)
        
        preds = np.concatenate([preds, outputs.cpu().detach().numpy()])
        reals = np.concatenate([reals, x.cpu().detach().numpy()])
    

    temp_preds=np.zeros((preds.shape[0]*jump_size, preds.shape[3]))
    temp_reals=np.zeros((preds.shape[0]*jump_size, preds.shape[3]))
    time_idx=0
    for i in range(len(preds)):
        temp_preds[time_idx:time_idx+jump_size, :] = preds[i, 0, :jump_size, :]
        temp_reals[time_idx:time_idx+jump_size, :] = reals[i, 0, :jump_size, :]
        time_idx += jump_size

    preds = temp_preds
    reals = temp_reals
    

    i=0
    num_per_row=4
    while(i < preds.shape[1]):
        fig, axs = plt.subplots(1, num_per_row, figsize=(15,5))

        for j in range(num_per_row):
            if i+j >= preds.shape[1]:
                break
            axs[j].plot(reals[:,i+j],alpha=.5,label='real')
            axs[j].plot(preds[:,i+j],alpha=.5,label='pred')
        
            axs[j].legend()
        plt.show()
        plt.close()

        i += j
        
    mse = mean_squared_error(reals, preds)
    print('MSE : ' + str(np.round(mse,10)))




def plot_reconstruction_prob_decoder(model, model_type, dataloader, X_tensor):
    model.eval()

    dataiter = iter(dataloader)
    x, y = dataiter.next()

    preds = np.empty((0,x.shape[1],x.shape[2],x.shape[3]))
    rec_mus = np.empty_like(preds)
    rec_sigmas = np.empty_like(preds)
    
    reals = np.empty((0,x.shape[1],x.shape[2],x.shape[3]))

    window_size = x.shape[2]
    cond_window_size = y.shape[2]
    jump_size = model.jump_size

    for j, data in enumerate(dataloader, 0):

        x, y = data
        x = x.cuda() if torch.cuda.is_available() else x.cpu()
        x.to(device)
        y = y.cuda() if torch.cuda.is_available() else y.cpu()
        y.to(device)
        if 'cvae' in model_type:
            outputs, rec_mu, rec_sigma, kl = model(x, y)
        else:
            outputs, rec_mu, rec_sigma, kl = model(x, None)
        
        preds = np.concatenate([preds, outputs.cpu().detach().numpy()])
        rec_mus = np.concatenate([rec_mus, rec_mu.cpu().detach().numpy()])
        rec_sigmas = np.concatenate([rec_sigmas, rec_sigma.cpu().detach().numpy()])
        
        reals = np.concatenate([reals, x.cpu().detach().numpy()])
    

    temp_preds=np.zeros((preds.shape[0]*jump_size, preds.shape[3]))
    temp_reals=np.zeros((preds.shape[0]*jump_size, preds.shape[3]))
    time_idx=0
    for i in range(len(preds)):
        temp_preds[time_idx:time_idx+jump_size, :] = preds[i, 0, :jump_size, :]
        temp_reals[time_idx:time_idx+jump_size, :] = reals[i, 0, :jump_size, :]
        time_idx += jump_size

    preds = temp_preds
    reals = temp_reals
    

    probs = []
    mu_to_plot = []#np.zeros_like(reals)
    sigma_to_plot = []#np.zeros_like(reals)
    for i in range(rec_mus.shape[0]):
        #for j in range(cond_window_size):
        for j in range(jump_size):

            mu_to_plot.append(rec_mus[i,0,j])
            sigma_to_plot.append(rec_sigmas[i,0,j])

            #probability of observed data point according to model
            prob = multivariate_normal.logpdf(X_tensor[i, 0, j], rec_mus[i,0,j], np.exp(rec_sigmas[i,0,j]))
            probs.append(prob)

    
    plt.figure(figsize=(20,6))
    plt.title('Log probs')
    plt.legend()
    plt.plot(probs)
    plt.show()

    mu_to_plot = np.array(mu_to_plot)
    sigma_to_plot = np.array(sigma_to_plot)


    '''
    for i in range(mu_to_plot.shape[1]):
        plt.figure(figsize=(15,5))
        plt.plot(reals[:, i],alpha=.5, label='real')
        plt.plot(mu_to_plot[:, i],alpha=.5, label='rec_mu')
        #plt.fill_between(np.arange(len(mu_to_plot[:, i])), mu_to_plot[:,i]-np.exp(sigma_to_plot[:,i]), mu_to_plot[:,i]+np.exp(sigma_to_plot[:,i]),alpha=0.2)

        #plt.plot(preds[:, i],alpha=.5, label='rec_mu')

        plt.legend()
        plt.show()
        plt.close()
    '''
    i=0
    num_per_row=4
    while(i < mu_to_plot.shape[1]):
        fig, axs = plt.subplots(1, num_per_row, figsize=(15,5))

        for j in range(num_per_row):
            if i+j >= mu_to_plot.shape[1]:
                break
            axs[j].plot(reals[:,i+j],alpha=.5,label='real')
            axs[j].plot(mu_to_plot[:,i+j],alpha=.5,label='pred')
        
            axs[j].legend()
        plt.show()
        plt.close()

        i += j


def read_machine_data(machine_name, window_size, jump_size, batch_size):
    
    X_train = pd.read_pickle(machine_name + '_train.pkl')
    X_test = pd.read_pickle(machine_name + '_test.pkl')
    Y_test = pd.read_pickle(machine_name + '_test_label.pkl')

    df_X_train, df_X_test, df_Y_test = pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(Y_test)

    X_train_data = df_X_train.values.copy()
    X_test_data = df_X_test.values.copy()

    #MSL data not 0-1... renormalize here...
    if 'MSL' in machine_name:
        scaler = MinMaxScaler()
        scaler.fit(X_train_data)
        X_train_data = scaler.transform(X_train_data)
        X_test_data = scaler.transform(X_test_data)

    #window it
    window = window_size
    jump = jump_size
    X_train = []
    X_test = []
    
    for i in range(0, X_train_data.shape[0]-window+1, jump):
        X_train.append([X_train_data[i:i+window]])

    for i in range(0, X_test_data.shape[0]-window+1, jump):
        X_test.append([X_test_data[i:i+window]])
        
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train_tensor = torch.from_numpy(X_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    
    train = torch.utils.data.TensorDataset(X_train_tensor, X_train_tensor)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

    test = torch.utils.data.TensorDataset(X_test_tensor, X_test_tensor)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    #print(X_train_tensor.shape, X_test_tensor.shape)
    
    return X_train_data, X_test_data, X_train_tensor, X_test_tensor, df_Y_test, trainloader, testloader

    

def read_machine_data_with_validation(machine_name, window_size, jump_size, batch_size, val_size=.3):
    
    X_train = pd.read_pickle(machine_name + '_train.pkl')
    X_test = pd.read_pickle(machine_name + '_test.pkl')
    Y_test = pd.read_pickle(machine_name + '_test_label.pkl')

    df_X_train, df_X_test, df_Y_test = pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(Y_test)

    X_train_data = df_X_train.values.copy()
    X_test_data = df_X_test.values.copy()

    #MSL data not 0-1... renormalize here...
    if 'MSL' in machine_name:
        scaler = MinMaxScaler()
        scaler.fit(X_train_data)
        X_train_data = scaler.transform(X_train_data)
        X_test_data = scaler.transform(X_test_data)

    #window it
    window = window_size
    jump = jump_size
    X_train = []
    X_test = []
    
    for i in range(0, X_train_data.shape[0]-window+1, jump):
        X_train.append([X_train_data[i:i+window]])

    for i in range(0, X_test_data.shape[0]-window+1, jump):
        X_test.append([X_test_data[i:i+window]])
        
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    X_train, X_val = train_test_split(X_train, test_size=val_size, shuffle=False)

    X_train_tensor = torch.from_numpy(X_train).float()
    X_val_tensor = torch.from_numpy(X_val).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    
    train = torch.utils.data.TensorDataset(X_train_tensor, X_train_tensor)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

    val = torch.utils.data.TensorDataset(X_val_tensor, X_val_tensor)
    valloader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)

    test = torch.utils.data.TensorDataset(X_test_tensor, X_test_tensor)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    #print(X_train_tensor.shape, X_test_tensor.shape)
    
    return X_train_data, X_test_data, X_train_tensor, X_test_tensor, df_Y_test, trainloader, valloader, testloader





    
def read_machine_data_cvae(machine_name, window_size, cond_window_size, jump_size, batch_size):
    
    X_train = pd.read_pickle(machine_name + '_train.pkl')
    X_test = pd.read_pickle(machine_name + '_test.pkl')
    Y_test = pd.read_pickle(machine_name + '_test_label.pkl')

    df_X_train, df_X_test, df_Y_test = pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(Y_test)

    X_train_data = df_X_train.values.copy()
    X_test_data = df_X_test.values.copy()

    #MSL data not 0-1... renormalize here...
    if 'MSL' in machine_name:
        scaler = MinMaxScaler()
        scaler.fit(X_train_data)
        X_train_data = scaler.transform(X_train_data)
        X_test_data = scaler.transform(X_test_data)


    #train data first
    window_size = window_size
    jump = jump_size
    X_train = []
    cond_train = []

    for i in range(0, X_train_data.shape[0]-cond_window_size-window_size+1, jump):
        X_train.append([X_train_data[i + cond_window_size : i + cond_window_size + window_size]])
        cond_train.append([X_train_data[i : i + cond_window_size]])
        
    X_train = np.array(X_train)
    cond_train = np.array(cond_train)
    X_train_tensor = torch.from_numpy(X_train).float()
    cond_train_tensor = torch.from_numpy(cond_train).float()

    train = torch.utils.data.TensorDataset(X_train_tensor, cond_train_tensor)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
        
    #test data now
    X_test = []
    cond_test = []

    X_test.append([X_test_data[0:window_size]])
    cond_test.append([X_train_data[-cond_window_size:]])
        
    for i in range(0, X_test_data.shape[0]-cond_window_size-window_size+1, jump):
        X_test.append([X_test_data[i+cond_window_size:i+cond_window_size+window_size]])
        cond_test.append([X_test_data[i:i+cond_window_size]])

    X_test = np.array(X_test)
    cond_test = np.array(cond_test)
    X_test_tensor = torch.from_numpy(X_test)
    cond_test_tensor = torch.from_numpy(cond_test)

    test = torch.utils.data.TensorDataset(X_test_tensor, cond_test_tensor)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    
    return X_train_data, X_test_data, X_train_tensor, cond_train_tensor, X_test_tensor, cond_test_tensor, df_Y_test, trainloader, testloader



def read_machine_data_cvae_with_validation(machine_name, window_size, cond_window_size, jump_size, batch_size, val_size=.3):
    
    X_train = pd.read_pickle(machine_name + '_train.pkl')
    X_test = pd.read_pickle(machine_name + '_test.pkl')
    Y_test = pd.read_pickle(machine_name + '_test_label.pkl')

    df_X_train, df_X_test, df_Y_test = pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(Y_test)

    X_train_data = df_X_train.values.copy()
    X_test_data = df_X_test.values.copy()

    #MSL data not 0-1... renormalize here...
    if 'MSL' in machine_name:
        scaler = MinMaxScaler()
        scaler.fit(X_train_data)
        X_train_data = scaler.transform(X_train_data)
        X_test_data = scaler.transform(X_test_data)


    #train data first
    window_size = window_size
    jump = jump_size
    X_train = []
    cond_train = []

    for i in range(0, X_train_data.shape[0]-cond_window_size-window_size+1, jump):
        X_train.append([X_train_data[i + cond_window_size : i + cond_window_size + window_size]])
        cond_train.append([X_train_data[i : i + cond_window_size]])
        
    X_train = np.array(X_train)
    cond_train = np.array(cond_train)

    X_train, X_val, cond_train, cond_val = train_test_split(X_train, cond_train, test_size=val_size, shuffle=False)

    X_train_tensor = torch.from_numpy(X_train).float()
    cond_train_tensor = torch.from_numpy(cond_train).float()
    X_val_tensor = torch.from_numpy(X_val).float()
    cond_val_tensor = torch.from_numpy(cond_val).float()

    train = torch.utils.data.TensorDataset(X_train_tensor, cond_train_tensor)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

    val = torch.utils.data.TensorDataset(X_val_tensor, cond_val_tensor)
    valloader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)
        
    #test data now
    X_test = []
    cond_test = []

    X_test.append([X_test_data[0:window_size]])
    cond_test.append([X_train_data[-cond_window_size:]])
        
    for i in range(0, X_test_data.shape[0]-cond_window_size-window_size+1, jump):
        X_test.append([X_test_data[i+cond_window_size:i+cond_window_size+window_size]])
        cond_test.append([X_test_data[i:i+cond_window_size]])

    X_test = np.array(X_test)
    cond_test = np.array(cond_test)
    X_test_tensor = torch.from_numpy(X_test).float()
    cond_test_tensor = torch.from_numpy(cond_test).float()

    test = torch.utils.data.TensorDataset(X_test_tensor, cond_test_tensor)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    
    return X_train_data, X_test_data, X_train_tensor, cond_train_tensor, X_test_tensor, cond_test_tensor, df_Y_test, trainloader, valloader, testloader
