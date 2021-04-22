import os
import argparse
import glob

import numpy as np

import torch

import utils
import evaluation_utils



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


def get_scores_and_labels_from_generation(model, df_Y_test, dataloader, X_test_tensor):
    
    preds, reals, scores, mse = evaluation_utils.evaluate_cvae_generation(model, dataloader, X_test_tensor)
    
    labels = df_Y_test.values
    labels = np.reshape(labels, (labels.shape[0], ))
    anomaly_idxs = np.where(labels == 1)[0]
    
    #create labels only up to num preds
    labels = np.zeros(len(scores), dtype=np.int)
    labels[anomaly_idxs] = 1

    return scores, labels, mse


def evaluate_models_from_folder(model_type, folder_path, batch_size, cvae_generation = False, start_from='1-1'):
    if cvae_generation and model_type != 'cvae':
        print('\nCannot generate with VAE, need CVAE to generate.\n')


    all_files = os.listdir(folder_path)
    all_files = sorted(all_files)

    tn_fp_fn_tp=np.empty((0,4)) 
    saved_models=[]
    for file in all_files:
        if file[-4:] == '.pth':
            saved_models.append(file)

    #start from code
    start_from = start_from +'.pth'
    start_idx = [i for i, s in enumerate(saved_models) if start_from in s]
    start_idx=start_idx[0]
    saved_models = saved_models[start_idx:]


    for model_file in saved_models:
        print(model_file)
        
        model = torch.load('{}/{}/{}'.format(os.getcwd(),folder_path,model_file))


        print(model_file)
        model_file = model_file.split(' ')[0]
        print(model_file)
        machine_name = model_file.split('-')
        machine_name = machine_name[1][-1] + '-' + machine_name[2].split('.')[0]

        print(machine_name)

        if 'vae' in model_type::
            X_train_data, X_test_data, X_train_tensor, X_test_tensor, df_Y_test, trainloader, testloader = utils.read_machine_data('../../datasets/ServerMachineDataset/machine-' + machine_name, model.window_size, model.jump_size, batch_size)
            
            scores, labels, mse = get_scores_and_labels(model_type, model, df_Y_test, testloader, X_test_tensor)

        if 'cvae' in model_type:
            X_train_data, X_test_data, X_train_tensor, cond_train_tensor, X_test_tensor, cond_test_tensor, df_Y_test, trainloader, testloader = utils.read_machine_data_cvae('../../datasets/ServerMachineDataset/machine-' + machine_name, model.window_size, model.cond_window_size, model.jump_size, batch_size)
            
            if cvae_generation:
                scores, labels, mse = get_scores_and_labels_from_generation(model, df_Y_test, testloader, X_test_tensor)
            else:
                scores, labels, mse = get_scores_and_labels(model_type, model, df_Y_test, testloader, X_test_tensor)


        confusion_matrix_metrics, alert_delays = evaluation_utils.compute_AUPR(labels, scores, threshold_jump=50)
        #confusion_matrix_metrics, alert_delays = evaluation_utils.compute_metrics_with_pareto(labels, scores, .01)
        print('[[TN, FP, FN, TP]]')
        print(confusion_matrix_metrics)
        print('Alert Delays : {}'.format(alert_delays))
        print('\n')

        tn_fp_fn_tp = np.concatenate([tn_fp_fn_tp, confusion_matrix_metrics])

        if cvae_generation:
            save_metrics_folder_path = folder_path + '/cvae_generation'
        else:
            save_metrics_folder_path = folder_path
        

        if not os.path.exists(save_metrics_folder_path):
            os.makedirs(save_metrics_folder_path)


        f = open('{}/{}/{}_confusion_metrics.txt'.format(os.getcwd(),save_metrics_folder_path,machine_name), "w")
        f.write('{} {} {} {}\n{}\n'.format(confusion_matrix_metrics[0,0],confusion_matrix_metrics[0,1],confusion_matrix_metrics[0,2],confusion_matrix_metrics[0,3], mse))
        for id, delay in enumerate(alert_delays):
            if id == 0:
                f.write('{}'.format(delay))
            else:
                f.write(' {}'.format(delay))
        f.close()

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


def get_combined_metrics(save_dir):
    txt_files = os.listdir(save_dir)
    txt_files = [i for i in txt_files if 'confusion_metrics.txt' in i]

    print('Number of metrics files : {} (should be 28)'.format(len(txt_files)))
    
    tn_fp_fn_tp=np.empty((0,4)) 

    for txt_file in txt_files:
        if 'confusion_metrics.txt' not in txt_file:
            continue
        else:
            f = open('{}/{}/{}'.format(os.getcwd(),save_dir,txt_file), 'r')
            
            lines = f.readlines()#[0]

            confusion_matrix_metrics = lines[0]
            confusion_matrix_metrics = np.array([int(i) for i in confusion_matrix_metrics.split(' ')]).reshape(1,4)
            tn_fp_fn_tp = np.concatenate([tn_fp_fn_tp, confusion_matrix_metrics])

            if len(lines) > 2:
                mse = float(lines[1])

                alert_delays = lines[2]
                alert_delays = alert_delays.split(' ')
                alert_delays = np.array([int(i) for i in alert_delays])


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

    model_type='cvae'
    save_dir = 'saved_models/cvae'
    cvae_generation=True

    #evaluate_models_from_folder(model_type, save_dir, 256, cvae_generation=cvae_generation, start_from='1-1')
    
    if not cvae_generation:
        get_combined_metrics(save_dir)
    else:
        get_combined_metrics(save_dir + '/cvae_generation')


if __name__ == '__main__':
    main()
