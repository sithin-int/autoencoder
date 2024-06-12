import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

import time as time
import copy as copy


# Autoencoder Architecture

class FC_Block(nn.Module):
    
    def __init__(self, in_feats, hidden_layers, activation_fn = nn.LeakyReLU()):
        
        super(FC_Block, self).__init__()
        
        layers = []
        
        for out_feats in hidden_layers:
            layers += [nn.Linear(in_feats, out_feats), activation_fn]
            in_feats = out_feats
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.block(x)
        
        return x


class Autoencoder(nn.Module):
    
    def __init__(self, input_dim, encoder_layers=[100,50,25], latent_dim=5, activation_fn = nn.LeakyReLU()):
        
        super(Autoencoder, self).__init__()
        
        self.encoder_block = FC_Block(input_dim, encoder_layers, activation_fn)
        
        self.embedding_layer = nn.Sequential(*[nn.Linear(encoder_layers[-1], latent_dim), activation_fn])
        
        decoder_layers = list(reversed(encoder_layers))
        self.decoder_block = FC_Block(latent_dim, decoder_layers, activation_fn)
        self.scores = nn.Linear(decoder_layers[-1], input_dim)
    
    def forward(self, x):
        
        x = self.encoder_block(x)
        h = x = self.embedding_layer(x)
        x = self.decoder_block(x)
        x = self.scores(x)
        
        return x, h


class bayesian_FC_Block(nn.Module):
    
    def __init__(self, in_feats, hidden_layers, activation_fn = nn.LeakyReLU(), dropout_prob=0.5):
        
        super(bayesian_FC_Block, self).__init__()
        
        layers = []
        
        for out_feats in hidden_layers:
            layers += [nn.Linear(in_feats, out_feats), activation_fn, nn.Dropout(p=dropout_prob)]
            in_feats = out_feats
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.block(x)
        
        return x


class bayesianAutoencoder(nn.Module):
    
    def __init__(self, input_dim, encoder_layers=[100,50,25], latent_dim=5, activation_fn = nn.LeakyReLU(), dropout_prob=0.5):
        
        super(bayesianAutoencoder, self).__init__()
        
        self.encoder_block = bayesian_FC_Block(input_dim, encoder_layers, activation_fn, dropout_prob)
        self.embedding_layer = nn.Sequential(*[nn.Linear(encoder_layers[-1], latent_dim), activation_fn, nn.Dropout(p=dropout_prob)])
        decoder_layers = list(reversed(encoder_layers))
        self.decoder_block = bayesian_FC_Block(latent_dim, decoder_layers, activation_fn, dropout_prob)
        self.scores = nn.Linear(decoder_layers[-1], input_dim)
    
    def forward(self, x):
        
        x = self.encoder_block(x)
        h = x = self.embedding_layer(x)
        x = self.decoder_block(x)
        x = self.scores(x)
        
        return x, h

# Training, Validation 

class Model:
    
    def __init__(self, net):
        self.net = net
        
    def compile(self, lr, h_lambda, loss_fn, cuda_device_id=0):
        
        self.h_lambda = h_lambda
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr)
        self.loss_fn = loss_fn 
        self.device = torch.device(f"cuda:{cuda_device_id}" if torch.cuda.is_available() else "cpu")
        
        self.net.to(self.device)
        
    def prepare_minibatch(self, mini_batch):
        
        inputs, targets = mini_batch
        
        return inputs.float().to(self.device), targets.float().to(self.device)
        
    def fit(self, dls, num_epochs, verbose=True):
        
        since = time.time()
        
        hist = {'train':{'loss':[]}, 'val':{'loss':[]}}
        
        best_loss = np.inf
        
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
                
                for mini_batch in dls[phase]:
                    
                    self.optimizer.zero_grad()
                    
                    inputs, targets = self.prepare_minibatch(mini_batch)
                    
                    with torch.set_grad_enabled(phase=="train"):
                        
                        recon_inputs, h = self.net(inputs)
                        
                        loss = self.loss_fn(recon_inputs, targets) + self.h_lambda * h.flatten().abs().sum()
                        
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
        
        return self.net.cpu()

# dataset definition
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, X):
        self.X = X
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.X[i]


# other useful utility functions

def norm_anomaly_split(X, y):
    
    normal_indeces = np.argwhere(y==0).ravel()
    anomaly_indeces = np.argwhere(y==1).ravel()
    
    X_norm = X[normal_indeces]
    X_anomaly = X[anomaly_indeces]

    return X_norm, X_anomaly

def visualize_using_tsne(X, y, n_components=2):
    
    X_transformed = TSNE(n_components = n_components, random_state=0).fit_transform(X)
    
    plt.scatter(*zip(*X_transformed[y==1]), marker='o', color='r', s=10, label='Anomalous')
    plt.scatter(*zip(*X_transformed[y==0]), marker='o', color='g', s=10, label='Normal')
    plt.legend()
    plt.show()
    
    
    
    