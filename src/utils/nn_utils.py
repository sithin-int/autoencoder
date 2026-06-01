import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time as time
import copy as copy


# Autoencoder Architecture

class FC_Block(nn.Module):
    
    def __init__(self, input_dim, hidden_dims):
        
        super(FC_Block, self).__init__()
        
        layers = []
        
        for output_dim in hidden_dims:
            layers += [nn.Linear(input_dim, output_dim), nn.LeakyReLU()]
            input_dim = output_dim
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):

        x = self.block(x)
        
        return x


class Autoencoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dims=[100,50,25], latent_dim=5):
        
        super(Autoencoder, self).__init__()
        
        self.encoder = FC_Block(input_dim, hidden_dims)
        self.embedding = nn.Sequential(*[nn.Linear(hidden_dims[-1], latent_dim), nn.LeakyReLU()])
        decoder_dims = list(reversed(hidden_dims))
        self.decoder = FC_Block(latent_dim, decoder_dims)
        self.output = nn.Linear(decoder_dims[-1], input_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        h = self.embedding(x)
        x = self.decoder(h)
        x = self.output(x)
        
        return x, h


class bayesian_FC_Block(nn.Module):
    
    def __init__(self, input_dim, hidden_dims, dropout_prob=0.5):
        
        super(bayesian_FC_Block, self).__init__()
        
        layers = []
        
        for output_dim in hidden_dims:
            layers += [nn.Linear(input_dim, output_dim), nn.LeakyReLU(), nn.Dropout(p=dropout_prob)]
            input_dim = output_dim
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.block(x)
        
        return x


class bayesianAutoencoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dims=[100,50,25], latent_dim=5, dropout_prob=0.5):
        
        super(bayesianAutoencoder, self).__init__()
        
        self.encoder = bayesian_FC_Block(input_dim, hidden_dims, dropout_prob)
        self.embedding = nn.Sequential(*[nn.Linear(hidden_dims[-1], latent_dim), nn.LeakyReLU(), nn.Dropout(p=dropout_prob)])
        decoder_dims = list(reversed(hidden_dims))
        self.decoder = bayesian_FC_Block(latent_dim, decoder_dims, dropout_prob)
        self.output = nn.Linear(decoder_dims[-1], input_dim)
    
    def forward(self, x):
        
        x = self.encoder(x)
        h = self.embedding(x)
        x = self.decoder(h)
        x = self.output(x)
        
        return x, h

# Training, Validation 

class Model:
    
    def __init__(self, net):
        self.net = net
        
    def compile(self, lr, l1_lambda, loss_fn, device):
        
        self.l1_lambda = l1_lambda
        self.loss_fn = loss_fn 
        self.device = device
        
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr)
        
    def prepare_minibatch(self, minibatch):
        
        inputs, targets = minibatch
        
        return inputs.float().to(self.device), targets.float().to(self.device)
        
    def fit(self, dls, num_epochs, verbose=True):
        
        since = time.time()
        
        hist = {'train':{'loss':[]}, 'val':{'loss':[]}}
        
        best_loss = np.inf

        best_model_wts = copy.deepcopy(self.net.state_dict())
        
        for epoch in range(num_epochs):
            
            if verbose:
                
                print('Epoch {}/{}'.format(epoch,num_epochs-1))
                print('-'*10)
                
            for phase in ["train", "val"]:
                
                if phase=="train":
                    self.net.train()
                else:
                    self.net.eval()
                    
                running_loss = 0.0
                
                for minibatch in dls[phase]:
                    
                    self.optimizer.zero_grad()
                    
                    inputs, _ = self.prepare_minibatch(minibatch)
                    
                    with torch.set_grad_enabled(phase=="train"):
                        
                        recon_inputs, h = self.net(inputs)
                        
                        loss = self.loss_fn(recon_inputs, inputs) + self.l1_lambda * h.abs().mean()
                        
                        if phase=="train":
                            
                            loss.backward()
                            self.optimizer.step()
                            
                        running_loss += loss.item()
                            
                epoch_loss = running_loss/len(dls[phase])
                hist[phase]["loss"].append(epoch_loss)
                
                if verbose:
                    print("{} Loss :{:.4f}".format(phase,epoch_loss))
                    
                if phase == "val":
                    
                    if epoch_loss<best_loss:
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(self.net.state_dict())
                        if verbose:
                            print(f"Checkpoing made at {epoch}")
                        
            if verbose:
                print()
                
            
        time_elapsed = time.time() - since
        
        if verbose:
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val Loss: {:4f}'.format(best_loss)) 

        
        self.net.load_state_dict(best_model_wts)


# other useful utility functions

def norm_anomaly_split(X, y):
    
    normal_indeces = np.argwhere(y==0).ravel()
    anomaly_indeces = np.argwhere(y==1).ravel()
    
    X_norm = X[normal_indeces]
    X_anomaly = X[anomaly_indeces]
    
    y_norm = y[normal_indeces]
    y_anomaly = y[anomaly_indeces]

    return X_norm, y_norm, X_anomaly, y_anomaly


    
    
    