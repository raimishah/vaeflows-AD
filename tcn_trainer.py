import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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

from early_stopping import EarlyStopping


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TCN_Trainer(nn.Module):

    def __init__(self, data_name, model_type, flow_type=None, early_stop_patience=5):
        super(TCN_Trainer, self).__init__()

        self.losses = []
        self.val_losses = []
        self.data_name = data_name
        self.model_type = model_type
        self.flow_type = flow_type
        self.es_train = EarlyStopping(patience=50)
        self.es_val = EarlyStopping(patience=early_stop_patience)

    def train_model(self, model, num_epochs, learning_rate, trainloader, valloader=None):
        
        self.flow_type = model.flow_type

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        tq = tqdm(range(num_epochs))

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=200, factor=.75)

        early_stopped_train=False
        early_stopped_val=False
        early_stopped=False

        val_flag=False
        flag = False
        for epoch in tq:
            #flag = False

            for j, data in enumerate(trainloader, 0):
                model.train()

                optimizer.zero_grad()

                #batches
                x, y = data
                x = x.cuda() if torch.cuda.is_available() else x.cpu()
                x.to(device)
                y = y.cuda() if torch.cuda.is_available() else y.cpu()
                y.to(device)

                if 'cvae' in self.model_type:
                    outputs, rec_mu, rec_sigma, kl = model(x, y)
                    if model.prob_decoder:
                        rec_comps, rec, rec_mu_sigma_loss, kl = model.loss_function(outputs[:, :, -model.jump_size:, :], x[:, :, -model.jump_size:, :], rec_mu[:, :, -model.jump_size:, :], rec_sigma[:, :, -model.jump_size:, :], kl)
                    else:
                        rec_comps, rec, rec_mu_sigma_loss, kl = model.loss_function(outputs[:, :, -model.jump_size:, :], x[:, :, -model.jump_size:, :], rec_mu, rec_sigma, kl)


                else:
                    outputs, rec_mu, rec_sigma, kl = model(x, None)
                    if model.prob_decoder:
                        rec_comps, rec, rec_mu_sigma_loss, kl = model.loss_function(outputs[:, :, -model.jump_size:, :], x[:, :, -model.jump_size:, :], rec_mu[:, :, -model.jump_size:, :], rec_sigma[:, :, -model.jump_size:, :], kl)
                    else:
                        rec_comps, rec, rec_mu_sigma_loss, kl = model.loss_function(outputs[:, :, -model.jump_size:, :], x[:, :, -model.jump_size:, :], rec_mu, rec_sigma, kl)


                #no prob decoder for this TCN
                #rec_comps, rec, rec_mu_sigma_loss, kl = model.loss_function(outputs[:, :, -model.jump_size:, :], x[:, :, -model.jump_size:, :], rec_mu[:, :, -model.jump_size:, :], rec_sigma[:, :, -model.jump_size:, :], kl)
                #rec_comps, rec, rec_mu_sigma_loss, kl = model.loss_function(outputs, x, rec_mu, rec_sigma, kl)

                loss = rec + kl + rec_mu_sigma_loss

                if(np.isnan(loss.item())):
                    #print("Noped out at", epoch, j, kl, rec_comps)
                    print("Noped out at epoch ", epoch)
                    flag = True
                    break


                if self.es_train.step(loss):
                    early_stopped_train=True


                loss.backward()

                #200k or so usually, sometimes 1mil
                #total_norm=0
                #for p in model.parameters():
                #    if p.grad is not None:
                #        param_norm = p.grad.data.norm(2)
                #        total_norm += param_norm.item() ** 2
                #total_norm = total_norm ** (1. / 2)
                #print(total_norm)
                
                clip_val = 1000000
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

                
                optimizer.step()



            if valloader != None:
                #VALIDATION
                with torch.no_grad():
                    model.eval()
                    for j, data in enumerate(valloader, 0):
                        x, y = data
                        x = x.cuda() if torch.cuda.is_available() else x.cpu()
                        x.to(device)
                        y = y.cuda() if torch.cuda.is_available() else y.cpu()
                        y.to(device)
                        if 'cvae' in self.model_type:
                            outputs, rec_mu, rec_sigma, kl = model(x, y)
                        else:
                            outputs, rec_mu, rec_sigma, kl = model(x, None)

                        #TODO
                        #DO REAL MSE INSTEAD OF LOSS_FUNCTION -- causing issue with early stopping I think
                        if model.prob_decoder:
                            
                            #no prob decoder implemented for tcn vae
                            #rec = torch.sum((outputs[:, :, -model.jump_size:, :] -  x[:, :, -model.jump_size:, :])**2)
                            rec = torch.sum((rec_mu - x) ** 2)
                        else:
                            rec = torch.sum((outputs[:, :, -model.jump_size:, :] -  x[:, :, -model.jump_size:, :])**2)
                            #rec = torch.sum((outputs - x) ** 2)

                        #_, rec, _, _ = model.loss_function(outputs, x, rec_mu, rec_sigma, kl)

                        val_loss = rec
                        if(np.isnan(val_loss.item())):
                            #print("Noped out in validation at", epoch, j, kl, rec_comps)
                            print("Nan in Validation at ", epoch)
                            flag = True
                            break

                        if self.es_val.step(val_loss):
                            early_stopped_val=True
                            #if early_stopped_train:
                            #    early_stopped=True
                            #    break
                            early_stopped=True
                            break

                        scheduler.step(val_loss)
                        

                self.val_losses.append(val_loss.item())
                scheduler.step(val_loss)

                
            if(flag) or early_stopped:
                break
            tq.set_postfix(loss=loss.item())

            self.losses.append(loss.item())
                
        return model, flag

    def plot_model_loss(self, save_dir, machine_name):
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,6))
        ax1.plot(self.losses,label='loss (total)', color='blue')
        ax2.plot(self.val_losses,label='validation loss (reconstruction only (MSE))', color='orange')
        plt.legend()
        if self.flow_type==None:
            plt.savefig(save_dir + str(self.model_type) + str(self.flow_type) + str(machine_name))
        else:
            plt.savefig(save_dir + str(self.model_type) + str(machine_name))
        plt.show()
