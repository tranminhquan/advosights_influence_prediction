import numpy as np
import pickle
import matplotlib.pyplot as plt

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
from torch_geometric.utils import train_test_split_edges

from utils.visualize import visualize_tsne

class VGAEEncoder(torch.nn.Module):
    def __init__(self, n_node_atts, emb_dim):
        super(VGAEEncoder, self).__init__()
        self.conv1 = GCNConv(n_node_atts, 2 * emb_dim)
        self.conv2 = GCNConv(2 * emb_dim, 4 * emb_dim)
        self.conv3 = GCNConv(4 * emb_dim, 4 * emb_dim)
        self.conv4 = GCNConv(4 * emb_dim, 2 * emb_dim)
        self.conv_mu = GCNConv(2 * emb_dim, emb_dim)
        self.conv_logvar = GCNConv(2 * emb_dim, emb_dim)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = F.leaky_relu(x)
        x = self.conv4(x, edge_index, edge_weight=edge_weight)
        x = F.leaky_relu(x)
        
        return self.conv_mu(x, edge_index, edge_weight=edge_weight), self.conv_logvar(x, edge_index, edge_weight=edge_weight)
    
class VGAEDecoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_weight):
        super(VGAEDecoder, self).__init__()
        
        self.edge_weight = edge_weight
        
        self.conv1 = GCNConv(in_dim, 2 * in_dim)
        self.conv2 = GCNConv(2 * in_dim, 2 * out_dim)
        self.conv3 = GCNConv(2 * out_dim, out_dim)
        
    def forward(self, x, edge_index, sigmoid):
        x = self.conv1(x, edge_index, edge_weight=self.edge_weight)
        x = self.conv2(x, edge_index, edge_weight=self.edge_weight)
        x = self.conv3(x, edge_index, edge_weight=self.edge_weight)
        
        if sigmoid:
            return F.sigmoid(x)
        return x

class VGAEEmb():
    def __init__(self, data, emb_dim=128, save_path=None, split_test=False,**kwargs):
        if type(data) is str:
            self.data = torch.load(data)
        elif type(data) is torch_geometric.data.Data:
            self.data = data
            
        self.dim = emb_dim
        self.save_path = save_path
        
        self.n_node_atts = self.data.x.shape[1]
        self.n_edge_atts = self.data.edge_attr.shape[1]
        
        self.scaler = sklearn.preprocessing.StandardScaler()
        
        # preprocessing
#         scale = kwargs['scale'] if 'scale' in kwargs else False
#         normalize = kwargs['normalize'] if 'normalize' in kwargs else False
#         print('- Preprocess graph data: ')
#         print('\t * Scale = ', scale)
#         print('\t * Normalize = ', normalize)
        
#         self.preprocess(scale=scale, normalize=normalize)
        self.data.x = torch.from_numpy(self.scaler.fit_transform(self.data.x))
        self.original_edge_index = self.data.edge_index

        # split test
        
        if split_test:
            self.data = train_test_split_edges(self.data)
        
            # split edge attr in data
            indices = self.get_edge_attr_indices(self.original_edge_index, self.data.train_pos_edge_index)
            train_pos_edge_attr = torch.zeros((len(indices), self.n_edge_atts), dtype=torch.float)
            for counter,i in enumerate(indices):
                if i != -1:
                    train_pos_edge_attr[counter] = self.data.edge_attr[i]

            self.data.train_pos_edge_attr = train_pos_edge_attr
            self.data.train_pos_edge_attr = self.cal_edge_weight(self.data.train_pos_edge_attr)
        
        self.data.edge_attr = self.cal_edge_weight(self.data.edge_attr)
        
        # set model
        self.model = torch_geometric.nn.VGAE(VGAEEncoder(self.n_node_atts, self.dim)) if not 'model' in kwargs else kwargs['model']
#         self.model = torch_geometric.nn.VGAE(VGAEEncoder(self.n_node_atts, self.dim), VGAEDecoder(self.dim, self.n_node_atts, edge_weight=))
        
        
#     def preprocess(self, scale, normalize):
#         # node attribute
#         if scale:
#             self.data.x = torch.from_numpy(sklearn.preprocessing.scale(self.data.x))
#         if normalize:
#             self.data.x = torch.from_numpy(sklearn.preprocessing.normalize(self.data.x))
            
    def cal_edge_weight(self, edge_attr):
        '''
        Function defines how to cal weight of edge from reactions, comments and shares
        - current version: f = #reactions + #comments + #shares / 3
        '''
#         self.data.edge_attr = torch.mean(self.data.edge_attr, axis=1)
        return (0.6*edge_attr[:,0] + 0.3*edge_attr[:,1] + 0.1*edge_attr[:,2])
#         self.data.train_pos_edge_attr = 0.6*self.data.train_pos_edge_attr[:,0] + 0.3*self.data.train_pos_edge_attr[:,1] + 0.1*self.data.train_pos_edge_attr[:,2]
        
    
    def train(self, epochs, device, optim='adam', **kwargs):
        split_test = False if not 'split_test' in kwargs or kwargs['split_test']==False else True
        return_optimal = False if not 'return_optimal' in kwargs or kwargs['return_optimal']== False else True
        
        if type(device) is str:
            if device == 'gpu' or device == 'cuda':
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cpu')

        self.model = self.model.to(device)
#         x, edge_index, edge_att = self.data.x.float().to(device), self.data.edge_index.to(device), self.data.edge_attr.float().to(device)        
        
        if split_test:   
            x, train_pos_edge_index = self.data.x.float().to(device), self.data.train_pos_edge_index.to(device)
            train_pos_edge_attr = self.data.train_pos_edge_attr.to(device)
        else:
            x, train_pos_edge_index = self.data.x.float().to(device), self.data.edge_index.to(device)
            train_pos_edge_attr = self.data.edge_attr.to(device)
        
        
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
            
            if split_test:
                auc, ap = self.test2(x, self.data.test_pos_edge_index, self.data.test_neg_edge_index,
                                train_pos_edge_index, train_pos_edge_attr)
                hauc.append(auc)
                hap.append(ap)
                print('- Epoch: {:02d}, Loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, loss, auc, ap))
            else:
                print('- Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss))
                
            hloss.append(loss)
            
            
            if not self.save_path is None:
                if monitor == 'ap' and split_test is True:
                    if ap > max_ap:
                        max_ap = ap
                        torch.save(self.model.state_dict(), self.save_path)
                        print('\t **Updated checkpoint**')
                        
                else:
                    if loss < min_loss:
                        min_loss = loss
                        torch.save(self.model.state_dict(), self.save_path)
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
        
        if return_optimal:
            optimal_info = {
                'n_node_atts': self.n_node_atts,
                'dim': self.dim,
                'weights': torch.load(self.save_path)
            }
            return [hloss, hauc, hap], optimal_info
        else:
            return [hloss, hauc, hap]
    
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
#         loss = self.model.recon_loss(z, edge_index)
        loss = self.model.recon_loss(z, edge_index)
        loss = loss + (1 / self.data.num_nodes) * self.model.kl_loss()
        
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
        if type(device) is str:
            if device == 'gpu' or device == 'cuda':
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cpu')
                
        if self.save_path is None:
            test_model = self.model
        else:
            test_model = torch.load(self.save_path)
          
        if data is None:
            data = self.data
        elif type(data) is str:
            data = torch.load(data)
            # print('Normalized')
            # data.x = torch.from_numpy(self.scaler.transform(data.x))
            # data.edge_attr = self.cal_edge_weight(data.edge_attr)
            
        elif type(data) is torch_geometric.data.Data:
            pass
            # print('Normalized')
            # data.x = torch.from_numpy(self.scaler.transform(data.x))
            # data.edge_attr = self.cal_edge_weight(data.edge_attr)      
            
        test_model = test_model.to(device)
        z = test_model.encode(data.x.float().to(device), data.edge_index.to(device), data.edge_attr.to(device))
        z = z.detach()

        if not save_emb_path is None:
            torch.save(z, save_emb_path)
                
        return z
    


    
