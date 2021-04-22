import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix

from scipy import stats
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import genpareto

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



'''
plot_error_and_anomaly_idxs(real, preds, scores, anomaly_idxs, thresh):
    inputs: real - real values
            preds - prediction values
            scores - score/probabilities of observation of real vals according to model
            anomaly_idxs - anomaly indices for testing
            thresh - threshold using for anomaly classification
    return: none
    
'''
def plot_error_and_anomaly_idxs(real, preds, scores, anomaly_idxs, thresh):

    plt.figure(figsize=(50,15))
    plt.plot(real)
    plt.plot(preds)
    for ai in anomaly_idxs:
        plt.plot(ai, 1)
    plt.show()
    
    plt.figure(figsize=(50,15))
    for idx,ai in enumerate(anomaly_idxs):
        plt.scatter(ai, scores[ai], color='red')
    plt.axhline(y=thresh, color='red', label='threshold')
    plt.plot(scores)
    plt.show()

    return




def get_anomaly_windows_i_j(labels):
    anomaly_windows = []
    i = 0
    while i < len(labels):
        if labels[i] == 1:
            j = i
            while(j < len(labels)):
                if labels[j] == 0:
                    anomaly_windows.append([i,j])
                    break
                j+=1

                if j == len(labels)-1 and labels[j] == 1:
                    anomaly_windows.append([i,j+1])
                    break                

            i = j-1

        i+=1
    return anomaly_windows


'''
evaluate_adjusted_anomalies(anomaly_windows, scores, thresh):
    inputs: anomaly_windows - from above function i,j indices of anomaly windows
            scores - score/probabilities of observation of real vals according to model
            thresh - threshold using for anomaly classification
    return: adjusted_alerts (predictions for anomalies based on DONUT method)
    
    description: computes anomaly based on DONUT method
    
'''
def evaluate_adjusted_anomalies(anomaly_windows, scores, thresh):
    pointwise_alerts = np.array([1 if scores[i] < thresh else 0 for i in range(len(scores))])

    adjusted_alerts = np.copy(pointwise_alerts)

    alert_delays = []

    for aw in anomaly_windows:
        if pointwise_alerts[aw[0]:aw[1]].any() == 1:
            #detection time
            first_alert_time = np.where(pointwise_alerts[aw[0]:aw[1]] == 1)[0][0]
            #print('\nDetection time: ')
            #print(first_alert_time)
            alert_delays.append(first_alert_time)
            



            adjusted_alerts[aw[0]:aw[1]] = 1
            

    alert_delays = np.array(alert_delays)

    return adjusted_alerts, alert_delays

'''
print_metrics(real, anomaly_preds):
    inputs: real - array of 0/1 for not anomaly, anomaly, respectively
            anomaly_preds - prediction 0/1 for not anomaly, anomaly, respectively
    return: none
'''
def print_metrics(labels, anomaly_preds):
    print('\n--- Metrics ---')
    precision = precision_score(labels, anomaly_preds)
    recall = recall_score(labels, anomaly_preds)
    f1 = f1_score(labels, anomaly_preds)
    print('precision : ' + str(precision) + ' recall : ' + str(recall) + ' f1 : ' + str(f1))
    print('\n')

    
    
'''
print_metrics(real, scores):
    inputs: real - array of 0/1 for not anomaly, anomaly, respectively
            scores - anomaly scores
    return: none
    description: plots PR curve and prints AUPR, best f1
'''    
def compute_AUPR(labels, scores, threshold_jump=5):

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    
    precisions = []
    recalls = []
    
    f1s = []
    
    print('Computing AUPR for {} thresholds ... '.format(len(thresholds[::threshold_jump])))    
    
    anomaly_windows = get_anomaly_windows_i_j(labels)
    for idx, th in enumerate(thresholds[::threshold_jump]):
        if idx == len(thresholds[::threshold_jump]) // 4 or idx == len(thresholds[::threshold_jump]) // 2 or idx == len(thresholds[::threshold_jump]) // (4/3):
            print(str(100*(idx / len(thresholds[::threshold_jump]))) +'% done')
        
        
        
        anomaly_preds, _ = evaluate_adjusted_anomalies(anomaly_windows, scores, th)
        precision = precision_score(labels, anomaly_preds)
        recall = recall_score(labels, anomaly_preds)
        f1 = f1_score(labels, anomaly_preds)
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
           
    plt.title('Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    plt.plot(recalls, precisions)
    plt.show()
    
    print('\n--- AUPR ---')
    print(auc(recalls, precisions))    

    best_f1_idx = np.argmax(f1s)
    best_f1_threshold = thresholds[best_f1_idx*threshold_jump]
    best_f1_score = f1s[best_f1_idx]
    
    best_AD_quantile = 1-stats.percentileofscore(-thresholds, np.abs(best_f1_threshold))/100

    print('Best F1 score : {} at threshold : {} (Best AD quantile : {})'.format(best_f1_score, best_f1_threshold, best_AD_quantile))
    print('Corresponding best precision : {}, best recall : {}'.format(precisions[best_f1_idx], recalls[best_f1_idx]))

    #return tn, fp, fn, tp
    anomaly_preds, alert_delays = evaluate_adjusted_anomalies(anomaly_windows, scores, best_f1_threshold)
    
    tn, fp, fn, tp = confusion_matrix(labels, anomaly_preds).ravel()
    
    return np.array([tn,fp,fn,tp]).reshape((1,4)), alert_delays

    
    
def compute_metrics_with_pareto(labels, scores, initial_quantile_thresh):

    initial_threshold = np.quantile(scores, initial_quantile_thresh)
    #print('initial thresh: ' + str(initial_threshold))

    tails = scores[scores < initial_threshold]
    #fit pareto to tail
    pareto = genpareto.fit(tails)
    c, loc, scale = pareto
    gamma = c
    beta = scale

    q = 10e-3 # desired probability to observe

    N_prime = len(scores)
    N_prime_th = (scores < initial_threshold).sum()
    #print(N_prime_th)

    final_threshold = initial_threshold - (beta/gamma) * ( np.power((q*N_prime)/N_prime_th, -gamma) - 1 )
    #print(final_threshold)
    #print('final thresh: ' + str(final_threshold))
    
    #plt.plot(scores)
    #plt.axhline(final_threshold, color='r')
    #plt.show()

    anomaly_windows = get_anomaly_windows_i_j(labels)
        
    anomaly_preds, alert_delays = evaluate_adjusted_anomalies(anomaly_windows, scores, final_threshold)

    precision = precision_score(labels, anomaly_preds)
    recall = recall_score(labels, anomaly_preds)
    f1 = f1_score(labels, anomaly_preds)
    
    print('F1 : {}, Precision : {}, Recall : {}'.format(f1, precision, recall))

    tn, fp, fn, tp = confusion_matrix(labels, anomaly_preds).ravel()

    return np.array([tn,fp,fn,tp]).reshape((1,4)), alert_delays








def evaluate_model_new(model, model_type, dataloader, X_tensor):
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
        elif 'sigma' in model_type:
            outputs, rec_mu, rec_sigma, kl = model(x)
        else:
            outputs, rec_mu, rec_sigma, kl = model(x, None)
        
        preds = np.concatenate([preds, outputs.cpu().detach().numpy()])
        if model.prob_decoder:
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

    nans = np.argwhere(np.isnan(preds))
    for nan in nans:
        preds[nan] = 1

    '''
    if model_type=='cvae':
        temp_preds=np.zeros((preds.shape[0]*cond_window_size, preds.shape[3]))
        temp_reals=np.zeros((reals.shape[0]*cond_window_size, reals.shape[3]))
        
        time_idx=0
        for i in range(len(preds)):
            temp_preds[time_idx:time_idx+cond_window_size, :] = preds[i, 0, :cond_window_size, :]
            temp_reals[time_idx:time_idx+cond_window_size, :] = reals[i, 0, :cond_window_size, :]
            time_idx += cond_window_size

        preds = temp_preds
        reals = temp_reals

    else:
        preds = np.reshape(preds, (preds.shape[0] * preds.shape[2], preds.shape[3]))
        reals = np.reshape(reals, (reals.shape[0] * reals.shape[2], reals.shape[3]))
    '''


    #get scores
    if model.prob_decoder:
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

        scores = np.array(probs)
    else:
        scores = - (np.square(preds - reals)).mean(axis=1)


    mse = mean_squared_error(reals, preds)
    print('MSE : ' + str(np.round(mse,10)))


    return preds, scores, mse


def evaluate_model_tcn(model, model_type, dataloader, X_tensor):
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
        if model.prob_decoder:
            rec_mus = np.concatenate([rec_mus, rec_mu.cpu().detach().numpy()])
            rec_sigmas = np.concatenate([rec_sigmas, rec_sigma.cpu().detach().numpy()])
        
        reals = np.concatenate([reals, x.cpu().detach().numpy()])
    

    temp_preds=np.zeros((preds.shape[0]*jump_size, preds.shape[3]))
    temp_reals=np.zeros((preds.shape[0]*jump_size, preds.shape[3]))
    time_idx=0
    for i in range(len(preds)):
        temp_preds[time_idx:time_idx+jump_size, :] = preds[i, 0, -jump_size: , :]
        temp_reals[time_idx:time_idx+jump_size, :] = reals[i, 0, -jump_size: , :]
        time_idx += jump_size

    preds = temp_preds
    reals = temp_reals

    nans = np.argwhere(np.isnan(preds))
    for nan in nans:
        preds[nan] = 1

    #get scores
    if model.prob_decoder:
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

        scores = np.array(probs)
    else:
        scores = - (np.square(preds - reals)).mean(axis=1)

    mse = mean_squared_error(reals, preds)
    print('MSE : ' + str(np.round(mse,10)))


    return preds, scores, mse









def evaluate_cvae_generation(model, dataloader, X_tensor):
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
    num_feats = X_tensor.shape[-1]


    for j, data in enumerate(dataloader, 0):

        x, y = data
        x = x.cuda() if torch.cuda.is_available() else x.cpu()
        x.to(device)
        y = y.cuda() if torch.cuda.is_available() else y.cpu()
        y.to(device)

        generated, rec_mu, rec_sigma, kl = model.generate(y)


        preds = np.concatenate([preds, generated.cpu().detach().numpy()])
        if model.prob_decoder:
            rec_mus = np.concatenate([rec_mus, rec_mu.cpu().detach().numpy()])
            rec_sigmas = np.concatenate([rec_sigmas, rec_sigma.cpu().detach().numpy()])
        
        reals = np.concatenate([reals, x.cpu().detach().numpy()])
    

    temp_preds=np.zeros((preds.shape[0]*jump_size, preds.shape[3]))
    temp_reals=np.zeros((reals.shape[0]*jump_size, reals.shape[3]))
        
    time_idx=0
    for i in range(len(preds)):
        temp_preds[time_idx:time_idx+jump_size, :] = preds[i, 0, :jump_size, :]
        temp_reals[time_idx:time_idx+jump_size, :] = reals[i, 0, :jump_size, :]
        time_idx += jump_size

    preds = temp_preds
    reals = temp_reals


    #get scores
    if model.prob_decoder:
        probs = []
        mu_to_plot = []#np.zeros_like(reals)
        sigma_to_plot = []#np.zeros_like(reals)
        for i in range(rec_mus.shape[0]):
            #for j in range(cond_window_size):
            for j in range(jump_size):

                mu_to_plot.append(rec_mus[i,0,j])
                sigma_to_plot.append(rec_mus[i,0,j])

                #probability of observed data point according to model
                prob = multivariate_normal.logpdf(X_tensor[i, 0, j], rec_mus[i,0,j], np.exp(rec_sigmas[i,0,j]))
                probs.append(prob)

        scores = np.array(probs)
    else:
        scores = - (np.square(preds - reals)).mean(axis=1)

    mse = mean_squared_error(reals, preds)
    print('MSE : ' + str(np.round(mse,10)))



    '''
    num_per_row = 5
    j = 0
    while j < num_feats:
        fig, axs = plt.subplots(1, num_per_row, figsize=(15,5))
        for k in range(num_per_row):
            if j+k >= num_feats:
                break

            axs[k].plot(reals[:, j+k], alpha=.5)
            if model.prob_decoder:
                axs[k].plot(mu_to_plot[:, j+k], alpha=.5)
            else:
                axs[k].plot(preds[:, j+k], alpha=.5)

        plt.show()
            
        j += k
    '''


    return preds, reals, scores, mse




















def evaluate_vae_model(model, X_tensor):
    '''
    X_tensor = X_tensor.cuda() if torch.cuda.is_available() else X_tensor.cpu()
    X_tensor.to(device)
    out_pred, _,_,_= model(X_tensor)
    out_pred = out_pred.cpu().detach().numpy()
        
    idx = 0
    preds=np.zeros((out_pred.shape[0]*out_pred.shape[2], out_pred.shape[3]))

    window_size = X_tensor.shape[2]
    
    time_idx=0
    for i in range(len(out_pred)):
        preds[time_idx:time_idx+window_size, :] = out_pred[i, 0, :window_size, :]
        time_idx += window_size

    return preds
    '''

def VAE_anomaly_detection(model, X_test_tensor, X_test_data, X_train_data, df_Y_test, initial_quantile_thresh):

    real = df_Y_test.values
    real = np.reshape(real, (real.shape[0], ))
    anomaly_idxs = np.where(real == 1)[0]
    
    #inference
    preds = evaluate_vae_model(model, X_test_tensor)
    scores = - (np.square(preds - X_test_data[:len(preds)])).mean(axis=1)

    #create real labels
    real = np.zeros(len(scores), dtype=np.int)
    real[anomaly_idxs] = 1
    
    compute_AUPR(real, scores)
    
    thresh = np.quantile(scores, initial_quantile_thresh)
    plot_error_and_anomaly_idxs(X_test_data, preds, scores, anomaly_idxs, thresh)
    anomaly_preds = evaluate_adjusted_anomalies(real, scores, thresh)
    print_metrics(real, anomaly_preds)

    
    
    

def evaluate_cvae_model(model, X_tensor, c):
    cond_window_size = c.size(2)
    
    X_tensor = X_tensor.cuda() if torch.cuda.is_available() else X_tensor.cpu()
    X_tensor.to(device)
    c = c.cuda() if torch.cuda.is_available() else c.cpu()
    c.to(device)

    out_pred, _,_,_= model(X_tensor, c)
    out_pred = out_pred.cpu().detach().numpy()

    preds=np.zeros((out_pred.shape[0]*cond_window_size, out_pred.shape[3]))
    time_idx=0
    for i in range(len(out_pred)):
        preds[time_idx:time_idx+cond_window_size, :] = out_pred[i, 0, :cond_window_size, :]
        time_idx += cond_window_size
    
    return preds


def cVAE_anomaly_detection(model, X_test_tensor, X_test_data, cond_test_tensor, X_train_data, df_Y_test, initial_quantile_thresh):

    cond_window_size = cond_test_tensor.shape[2]

    #inference
    preds = evaluate_cvae_model(model, X_test_tensor, cond_test_tensor)
    scores = - (np.square(preds - X_test_data[:len(preds)])).mean(axis=1)

    real = df_Y_test.values
    real = np.reshape(real, (real.shape[0], ))
    real = real[:len(preds)]
    anomaly_idxs = np.where(real == 1)[0]
    
    compute_AUPR(real, scores)
    
    thresh = np.quantile(scores, initial_quantile_thresh)
    plot_error_and_anomaly_idxs(X_test_data, preds, scores, anomaly_idxs, thresh)
    anomaly_preds = evaluate_adjusted_anomalies(real, scores, thresh)
    print_metrics(real, anomaly_preds)