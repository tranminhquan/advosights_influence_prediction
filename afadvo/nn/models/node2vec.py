import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils.visualize import visualize_tsne

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch.utils.data import DataLoader
from collections import OrderedDict
import torch_geometric.visualization
import time

import torch_geometric.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from sklearn.manifold import TSNE
import sklearn
from torch_geometric.nn import GCNConv, GAE, VGAE

class Node2VecEmb():
    '''
    Node embedding using Node2Vec algorithm implemented by Torch-Geometric
    '''
    
    def __init__(self, data, emb_dim=128, walk_length=20, context_size=10, walks_per_node=10, save_path=None, **kwargs):
        '''
        Initialize for Node2Vec algorithm
        ---
        + data: Data torch-geometric or str. If str it should be path to file
        + emb_dim: emebdding dimension
        + walk_length: length of node to walk in RandomWalk algorithm
        + context_size: size of windows in skip-gram model
        + walks_per_node
        + save_path
        '''
        
        if type(data) is str:
            self.data = torch.load(data)
        elif type(data) is torch_geometric.data.Data:
            self.data = data
            
        self.dim = emb_dim
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.save_path = save_path
        
        # define Node2Vec algorith from torch-geometric
        self.model = torch_geometric.nn.Node2Vec(num_nodes = self.data.num_nodes, embedding_dim = self.dim, walk_length = self.walk_length,
                             context_size = self.context_size, walks_per_node = self.walks_per_node)
        
    def train(self, batch_size, epochs, device, optim='adam', shuffle=False, **kwargs):
        '''
        Train a Node2Vec model
        '''
        if type(device) is str:
            if device == 'gpu' or device == 'cuda':
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cpu')

         # set device
        self.model, self.data = self.model.to(device), self.data.to(device)
        
        loader = DataLoader(torch.arange(self.data.num_nodes), batch_size = batch_size, shuffle = shuffle)
        
        lr = kwargs['lr'] if 'lr' in kwargs else 0.001
        if optim == 'adam':
            optim = torch.optim.Adam(self.model.parameters(), lr=lr)
            
        print('Training Node2Vec with batch_size=', batch_size, ', epochs= ', epochs, ', optim=', optim, '\n -----------------')
        
        if device.type == 'cuda':
            print('*GPU is activated')
        else:
            print('*CPU is activated')
        
        # train
        start_time = time.time()
        hloss = []
        min_loss = 1000.
        
        for epoch in range(epochs):
            loss = self.single_train(loader, optim, device)
            hloss.append(loss)
            
            print('- Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss))
            
            if not self.save_path is None:
                if loss < min_loss:
                    min_loss = loss
                    torch.save(self.model, self.save_path)
                    print('\t **Updated checkpoint**')
                    
        print('Training complete.')
        print('\n------------------------------\n')
        print('Training time: ', time.time() - start_time)
        
        if not self.save_path is None:
            print('Minimum loss: ', min_loss)
        else:
            print('Latest loss: ', loss)
        
        return hloss
        
        
    def single_train(self, loader, optim, device):
        '''
        Train Node2Vec in a single epoch
        '''
        self.model.train()
        total_loss = 0.0
        
        for i, subset in enumerate(loader):
            optim.zero_grad()
            loss = self.model.loss(self.data.edge_index, subset.to(device))
            loss.backward()
            optim.step()
            
            total_loss += loss.item()
            
        return total_loss / len(loader)
    
    def predict(self, node_ids, device, save_emb_path=None):
        '''
        Embed a torch tensor of node ids to vector spaced using trained model
        '''
        if self.save_path is None:
            test_model = self.model
        else:
            test_model = torch.load(self.save_path)
            
        node_ids = torch.from_numpy(np.array(node_ids))
        node_ids = node_ids.to(device)
        test_model = test_model.to(device)
        
        z = test_model(node_ids)
        if not save_emb_path is None:
            with open(save_emb_path, 'wb') as dt:
                pickle.dump(z, dt)
                
        return z
    
    def visualize(self, z, algorithm='tsne', n_dim=2, figsize=(150,150), save_emb_path=None, save_fig_path=None):
        '''
        Visualize vector space by given alogrithm
        '''
        if type(z) is str:
            with open(z, 'rb') as dt:
                z = pickle.load(dt)
            
        visualize_tsne(z, n_dim, figsize, save_emb_path, save_fig_path)