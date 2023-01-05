import numpy as np
import pickle
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, train_test_split_edges
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

from utils.visualize import visualize_tsne

class GAEEncoder(torch.nn.Module):
    def __init__(self, n_node_atts, emb_dim):
        super(GAEEncoder, self).__init__()
        self.conv1 = GCNConv(n_node_atts, 2 * emb_dim)
        self.conv2 = GCNConv(2 * emb_dim, 4 * emb_dim)
        self.conv3 = GCNConv(4 * emb_dim, 4 * emb_dim)
        self.conv4 = GCNConv(4 * emb_dim, 2 * emb_dim)
        self.conv5 = GCNConv(2 * emb_dim, emb_dim)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = F.leaky_relu(x)
        x = self.conv4(x, edge_index, edge_weight=edge_weight)
        x = F.leaky_relu(x)
        x = self.conv5(x, edge_index, edge_weight=edge_weight)
        return x
    
class GAEEmb():
    def __init__(self, data, emb_dim=128, save_path=None, **kwargs):
        if type(data) is str:
            self.data = torch.load(data)
        elif type(data) is torch_geometric.data.Data:
            self.data = data
            
        self.dim = emb_dim
        self.save_path = save_path
        
        self.n_node_atts = self.data.x.shape[1]
        self.n_edge_atts = self.data.edge_attr.shape[1]
        
        self.scaler = sklearn.preprocessing.StandardScaler()

#         self.preprocess(scale=scale, normalize=normalize)
        self.data.x = torch.from_numpy(self.scaler.fit_transform(self.data.x))
        self.original_edge_index = self.data.edge_index

        # split test
        self.data = train_test_split_edges(self.data)
        
        # split edge attr in data
        indices = self.get_edge_attr_indices(self.original_edge_index, self.data.train_pos_edge_index)
        train_pos_edge_attr = torch.zeros((len(indices), self.n_edge_atts), dtype=torch.float)
        for counter,i in enumerate(indices):
            if i != -1:
                train_pos_edge_attr[counter] = self.data.edge_attr[i]
        self.data.train_pos_edge_attr = train_pos_edge_attr
     
        
        self.data.edge_attr = self.cal_edge_weight(self.data.edge_attr)
        self.data.train_pos_edge_attr = self.cal_edge_weight(self.data.train_pos_edge_attr)
        
        # set model
        self.model = torch_geometric.nn.GAE(GAEEncoder(self.n_node_atts, self.dim))

            
    def cal_edge_weight(self, edge_attr):
        '''
        Function defines how to cal weight of edge from reactions, comments and shares
        - current version: f = #reactions + #comments + #shares / 3
        '''
#         self.data.edge_attr = torch.mean(self.data.edge_attr, axis=1)
        return (0.6*edge_attr[:,0] + 0.3*edge_attr[:,1] + 0.1*edge_attr[:,2])
#         self.data.train_pos_edge_attr = 0.6*self.data.train_pos_edge_attr[:,0] + 0.3*self.data.train_pos_edge_attr[:,1] + 0.1*self.data.train_pos_edge_attr[:,2]
        
    
    def train(self, epochs, device, optim='adam', **kwargs):

        if type(device) is str:
            if device == 'gpu' or device == 'cuda':
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cpu')
        
        self.model = self.model.to(device)
#         x, edge_index, edge_att = self.data.x.float().to(device), self.data.edge_index.to(device), self.data.edge_attr.float().to(device)        
        
           
        x, train_pos_edge_index = self.data.x.float().to(device), self.data.train_pos_edge_index.to(device)
        train_pos_edge_attr = self.data.train_pos_edge_attr.to(device)
        
        
        lr = kwargs['lr'] if 'lr' in kwargs else 0.001
        if optim == 'adam':
            optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optim == 'sgd':
            optim = torch.optim.SGD(self.model.parameters(), lr=lr)
            
        monitor = kwargs['monitor'] if 'monitor' in kwargs else 'loss'
        
        print('Training GAE with epochs= ', epochs, ', optim=', optim, '\n -----------------')
        
        if device.type == 'cuda':
            print('*GPU is activated')
        else:
            print('*CPU is activated')
        
        # train
        start_time = time.time()
        hloss = []
        hauc = []
        hap = []
        min_loss = 1000.
        max_ap = 0.0
        
        for epoch in range(epochs):

            loss = self.single_train(x, train_pos_edge_index, train_pos_edge_attr, optim, device)
            
            auc, ap = self.test2(x, self.data.test_pos_edge_index, self.data.test_neg_edge_index,
                                train_pos_edge_index, train_pos_edge_attr)
#             auc=0.
#             ap=0.
            hloss.append(loss)
            hauc.append(auc)
            hap.append(ap)
            
            print('- Epoch: {:02d}, Loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, loss, auc, ap))
            
            if not self.save_path is None:
                if monitor == 'loss':
                    if loss < min_loss:
                        min_loss = loss
                        torch.save(self.model, self.save_path)
                        print('\t **Updated checkpoint**')
                elif monitor == 'ap':
                    if ap > max_ap:
                        max_ap = ap
                        torch.save(self.model, self.save_path)
                        print('\t **Updated checkpoint**')
                    
        print('Training complete.')
        print('\n------------------------------\n')
        print('Training time: ', time.time() - start_time)
        
        if not self.save_path is None:
            if monitor == 'loss':
                print('Minimum loss: ', min_loss)
            elif monitor == 'ap':
                print('Maximun AP: ', max_ap)
        else:
            print('Latest loss: ', loss)
        
        return hloss, hauc, hap
    
    def get_edge_attr_indices(self, origin_edge_index, edge_index):
        indices = []
        for i in range(edge_index.shape[1]):
            index = torch.where((origin_edge_index[0,:] == edge_index[0, i]) & (origin_edge_index[1,:] == edge_index[1, i]))[0]
            if len(index) == 0:
                indices.append(-1)
            else:
                indices.append(int(index[0]))
        return indices
    
    
    def single_train(self, x, edge_index, edge_att, optim, device):
        '''
        Train Node2Vec in a single epoch
        '''
        self.model.train()
        optim.zero_grad()
        
        z = self.model.encode(x, edge_index, edge_att)
        loss = self.model.recon_loss(z, edge_index)        
        loss.backward()
        optim.step()
        
        return loss.item()
    
    def single_train2(self, x, edge_index, optim, device):
        '''
        Train Node2Vec in a single epoch
        '''
        self.model.train()
        optim.zero_grad()
        
        z = self.model.encode(x, edge_index)
        loss = self.model.recon_loss(z, edge_index)
        loss.backward()
        optim.step()
        
        return loss.item()
    
    def test(self, x, pos_edge_index, neg_edge_index, train_pos_edge_index):
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(x, train_pos_edge_index)
        return self.model.test(z, pos_edge_index, neg_edge_index)
    
    def test2(self, x, pos_edge_index, neg_edge_index, train_pos_edge_index, train_pos_edge_attr):
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(x, train_pos_edge_index, train_pos_edge_attr)
        return self.model.test(z, pos_edge_index, neg_edge_index)
        
    def predict(self, data, device, save_emb_path=None, **kwargs):
        if self.save_path is None:
            test_model = self.model
        else:
            test_model = torch.load(self.save_path)
          
        if data is None:
            data = self.data
        elif type(data) is str:
            data = torch.load(data)
            print('Normalized')
            data.x = torch.from_numpy(self.scaler.transform(data.x))
            data.edge_attr = self.cal_edge_weight(data.edge_attr)
            
        elif type(data) is torch_geometric.data.Data:
            print('Normalized')
            data.x = torch.from_numpy(self.scaler.transform(data.x))
            data.edge_attr = self.cal_edge_weight(data.edge_attr)      
            
        test_model = test_model.to(device)
        z = test_model.encode(data.x.float().to(device), data.edge_index.to(device), data.edge_attr.to(device))
        
        if not save_emb_path is None:
            with open(save_emb_path, 'wb') as dt:
                pickle.dump(z, dt)
                
        return z
