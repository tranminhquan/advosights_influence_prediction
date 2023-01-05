import math
import numpy as np
import time
import pickle

import torch
import torch.nn as nn
from torch.optim import *

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from nn.models.hiersoftmax import HierSoftmax, JointHierSoftmax

class CSIF(nn.Module):
    def __init__(self, dim, upair_dim, post_dim, n_unique_upairs, n_unique_posts,hs_weights=None):
        
        '''
        Args:
            triplets: ndarray or str (path). Shape of (n_data x (user_embedding*2 + post_embedding))
            upair_dim: dimension of user pairs in triplets
            triplet_dict: dictionary or str (path). Keys are user id pairs and values are post ids
            dim: int, embedding dimension
        '''
        
        super(CSIF, self).__init__()
        
        self.dim = dim
        self.upair_dim = upair_dim
        self.post_dim = post_dim
        
        # randomly init projection matrices U and T
        self.U, self.T = self.init_weights(self.dim, self.upair_dim, self.post_dim)
        
        # hierarchical softmax
        self.hs = JointHierSoftmax(self.dim, n_unique_upairs, self.dim, n_unique_posts)
        if not hs_weights is None:
            print('Load hierarchical softmax weights')
            self.hs.load_state_dict(torch.load(hs_weights))
        
        self.hs.eval()
        
    def init_weights(self, dim, upair_dim, post_dim):
        return nn.Parameter(torch.rand(dim, self.upair_dim, requires_grad=True)), nn.Parameter(torch.rand(dim, self.post_dim, requires_grad=True))
    
    def forward(self, u_ij, u_indices, t_k, t_indices, device):
        
        x1 = torch.mm(self.T, t_k).T
        x2 = torch.mm(self.U, u_ij).T
        
#         # normalize
        x1 = self.normalize_data(x1)
        x2 = self.normalize_data(x2)
        
        x, _, _ = self.hs((x1, u_indices), (x2, t_indices), device)

        return x
    
    def normalize_data(self, tensor, dim=1):

        tensor = tensor - tensor.mean()
        tensor = tensor / tensor.std()
        
        return tensor
    
    def predict(self, u_ij, t_k, device=torch.device('cuda')):
        self.eval()
        
        u_ij = torch.from_numpy(u_ij) if type(u_ij) is np.ndarray else u_ij
        t_k = torch.from_numpy(t_k) if type(t_k) is np.ndarray else t_k
        
        self = self.to(device)
        u_ij = u_ij.to(device)
        u_ij = u_ij.unsqueeze(-1).float()
        
        t_k = t_k.to(device)
        t_k = t_k.unsqueeze(-1).float()
        
        u = torch.mm(self.U, u_ij).T
        v = torch.mm(self.T, t_k)
        
        u = self.normalize_data(u, 1)
        v = self.normalize_data(v, 0)

        return torch.mm(u,v) / math.sqrt(self.dim), u, v


class AF_scoring_v2:
    '''
    Content Social Influence embedding
    '''
    
    def __init__(self, triplets, upair_dim, triplet_dict, dim, test_size=0.3, **kwargs):
        '''
        Args:
            triplets: ndarray or str (path). Shape of (n_data x (user_embedding*2 + post_embedding))
            upair_dim: dimension of user pairs in triplets
            triplet_dict: dictionary or str (path). Keys are user id pairs and values are post ids
            dim: int, embedding dimension
        '''
        
        self.triplets = pickle.load(open(triplets, 'rb')) if type(triplets) is str else triplets
        
        self.triplets = self.normalize_data(self.triplets)

        self.triplet_dict = pickle.load(open(triplet_dict, 'rb')) if type(triplet_dict) is str else triplet_dict
        self.dim = dim
        
        # get unique triplets for hierarchical softmax
        self.unique_upairs, self.unique_posts = self._get_unique_triplets(self.triplet_dict)
        
        # unique to indices
        self.unique_upairs_dict = dict(zip(self.unique_upairs, np.arange(len(self.unique_upairs))))
        self.unique_posts_dict = dict(zip(self.unique_posts, np.arange(len(self.unique_posts))))
        
        self.upair_dim = upair_dim
        self.post_dim = self.triplets.shape[1] - self.upair_dim
        
        
        # hierarchical softmax
        self.hs = JointHierSoftmax(self.dim, len(self.unique_upairs), self.dim, len(self.unique_posts))
        
        # split train / test
        rd_state = kwargs['random_state'] if 'random_state' in kwargs else 121
        
        self.triplets, self.test_triplets, self.triplet_dict, self.test_triplet_dict = train_test_split(self.triplets, np.array([[k,v] for k,v in self.triplet_dict.items()]), test_size=test_size, random_state=rd_state)
        self.triplet_dict = dict(zip(self.triplet_dict[:,0], self.triplet_dict[:,1]))
        self.test_triplet_dict = dict(zip(self.test_triplet_dict[:,0], self.test_triplet_dict[:,1]))
        
        # generate indices for user_pair and post by indices
        
        self.upair_dict = dict(zip(np.arange(len(self.triplets)), self.triplet_dict.keys()))
        self.post_dict = dict(zip(np.arange(len(self.triplets)), self.triplet_dict.values()))
        self.test_upair_dict = dict(zip(np.arange(len(self.test_triplets)), self.test_triplet_dict.keys()))
        self.test_post_dict = dict(zip(np.arange(len(self.test_triplets)), self.test_triplet_dict.values()))
    
        # transpose triplets if not fit right shape
        self.triplets = self._transform_tripets(self.triplets, self.triplet_dict)
        self.test_triplets = self._transform_tripets(self.test_triplets, self.test_triplet_dict)
        
        print('Split data into train/test: ', self.triplets.shape, ' / ', self.test_triplets.shape)
        
        
    def _transform_tripets(self, triplets, triplet_dict):
        triplets = torch.from_numpy(triplets).float() if type(triplets) is np.ndarray else triplets.float()
        
        # normalize
#         triplets = self.normalize_data(triplets)
        
        if triplets.shape[1] != len(triplet_dict):
            triplets = triplets.T
        return triplets
        
    def _get_unique_triplets(self, triplet_dict):
        '''
        get unique sample for user pairs and post from triplet_dict (for hierarchical softmax)
        '''
        unique_upairs = np.unique(np.array(list(triplet_dict.keys())))
        unique_posts = np.unique(np.array(list(triplet_dict.values())))
        
        return unique_upairs, unique_posts
        
    def normalize_hiersoftmax_input(self, triplets):
        # scale triplets to [0,1]
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        triplets = min_max_scaler.fit_transform(triplets)
        triplets = torch.from_numpy(triplets)
        
        return triplets
    
    def normalize_data(self, tensor):
#         tensor -= tensor.min(1, keepdim=True)[0]
#         tensor /= tensor.max(1, keepdim=True)[0]
        
        tensor = tensor - tensor.mean()
        tensor = tensor / tensor.std()
        
        return tensor
    
    def test_hiersoftmax(self, batch_size, device, **kwargs):
        steps = len(self.test_triplet_dict) // batch_size
        batch_indices = []
        
        for i in range(steps):
            batch_indices.append(np.arange(i*batch_size, (i+1)*batch_size))
        if (len(self.triplet_dict)) % batch_size != 0:
            batch_indices.append(np.arange((steps * batch_size), (steps * batch_size)  + (len(self.test_triplet_dict) % batch_size)))
            steps = steps + 1
        
        loss = 0.0
        
        print('Test model')
        
        if str(device) == 'cuda':
            self.hs.cuda()
        else:
            self.hs.cpu()
            
        self.hs.eval()
        
        for i,indices in enumerate(batch_indices):
            batch_triplets = self.test_triplets[:, indices]
            
            t_outputs = [self.test_post_dict[k] for k in indices]
            u_outputs = [self.test_upair_dict[k] for k in indices]
            
            u_indices = [self.unique_upairs_dict[u] for u in u_outputs]
            t_indices = [self.unique_posts_dict[t] for t in t_outputs]
                
            t_k = batch_triplets[-self.post_dim:].to(device)
            u_ij = batch_triplets[:self.upair_dim].to(device)
            
            _u = torch.mm(self.T, t_k).T
            _t = torch.mm(self.U, u_ij).T
            
#             _u = _u / math.sqrt(self.dim)
#             _t = _t / math.sqrt(self.dim)
            
            _u = self.normalize_data(_u)
            _t = self.normalize_data(_t)
                        
            _loss, _, _ = self.hs((_u, u_indices), (_t, t_indices), device)
            
            loss += _loss.item()
            
        return loss / steps
    
    def get_batch(self, triplets, indices):
        
        batch_triplets = triplets[:, indices]

        t_outputs = [self.post_dict[k] for k in indices]
        u_outputs = [self.upair_dict[k] for k in indices]
        
        u_indices = [self.unique_upairs_dict[u] for u in u_outputs]
        t_indices = [self.unique_posts_dict[t] for t in t_outputs]

        t_k = batch_triplets[-self.post_dim:]
        u_ij = batch_triplets[:self.upair_dim]
        
        return u_ij, t_k
        
    def train_batch(self, i, model, triplets, indices, optim, device, model_type, verbose=False):
        
        start_batch = time.time()
        
        batch_triplets = triplets[:, indices]

        t_outputs = [self.post_dict[k] for k in indices]
        u_outputs = [self.upair_dict[k] for k in indices]
        
        u_indices = [self.unique_upairs_dict[u] for u in u_outputs]
        t_indices = [self.unique_posts_dict[t] for t in t_outputs]

        t_k = batch_triplets[-self.post_dim:].to(device)
        u_ij = batch_triplets[:self.upair_dim].to(device)
        
        
        optim.zero_grad()
        
        start_loss = time.time()
        if model_type == 'hiersoftmax':
            _u = torch.mm(self.T, t_k).T
            _t = torch.mm(self.U, u_ij).T
            
#             _u = _u / math.sqrt(self.dim)
#             _t = _t / math.sqrt(self.dim)
            
            _u = self.normalize_data(_u)
            _t = self.normalize_data(_t)
            
            _loss, _a, _b = model((_u, u_indices), (_t, t_indices), device)
            
            
        elif model_type == 'projection':
            _a, _b, _u, _t = None, None, None, None
            
            _loss = model(u_ij, u_indices, t_k, t_indices, device)
            
           
        if verbose:
            print('\t - Step ', i, '\t loss: ', _loss.item())

        start_backward = time.time()
        _loss.backward()
#         print('Backward: ', time.time() - start_backward)
        
        start_step = time.time()
        optim.step()
#         print('Optimze: ', time.time() - start_step)
        
        
        return _loss, _a, _b, _u, _t
        
    def train_hiersoftmax(self, optim, batch_size, epochs, device, save_path=None, shuffle=True, **kwargs):
        '''
        training Auxalaries Matrics hierarchical softmax (phase 1)
        '''
        
        steps = len(self.triplet_dict) // batch_size
        batch_indices = []
        
        for i in range(steps):
            batch_indices.append(np.arange(i*batch_size, (i+1)*batch_size))
        if (len(self.triplet_dict)) % batch_size != 0:
            batch_indices.append(np.arange((steps * batch_size), (steps * batch_size)  + (len(self.triplet_dict) % batch_size)))
            steps = steps + 1
        
        print('Training hierarchical softmax on ', len(self.triplet_dict), ' samples in ', steps, ' steps')
        
        print('Set tensor to gpu')
        self.U = torch.rand(self.dim, self.upair_dim).to(device)
        self.T = torch.rand(self.dim, self.post_dim).to(device)
        self.triplets = self.triplets.float().to(device)
        
        # load weight
        if 'weights' in kwargs:
            self.hs.load_state_dict(torch.load(kwargs['weights']))
        
        if str(device) == 'cuda':
            self.hs.cuda()
        else:
            self.hs.cpu()
            
        self.hs.train()
        
        lr = kwargs['lr'] if 'lr' in kwargs else 0.01
        weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs else 0
        
        if optim == 'adam':
            optim = Adam(self.hs.parameters(), lr=lr, weight_decay=weight_decay)
        
        elif optim == 'adagrad':
            optim = Adagrad(self.hs.parameters(), lr=lr, weight_decay=weight_decay)
        
        elif optim == 'sgd':
            optim = SGD(self.hs.parameters(), lr=lr, weight_decay=weight_decay)
        
        hloss = []
        min_loss = 1000.
        for epoch in range(epochs):
            if shuffle:
                temp_triplets = self.triplets[:, torch.randperm(self.triplets.size()[1])]
            
            u_loss = 0.0
            t_loss = 0.0
            loss = 0.0
            print('- Epoch ', epoch)
            
            for i,indices in enumerate(batch_indices):
                _loss, _a, _b, _u, _t = self.train_batch(i, self.hs, temp_triplets, indices, optim, device, model_type='hiersoftmax')
                
#                 return _a, _b, _u, _t
        
                loss += _loss.item()
    
            print('\t loss: ', loss / steps)
            
            test_loss = self.test_hiersoftmax(batch_size=batch_size, device=device)
            print('\t test loss: ', test_loss)
            
            hloss.append([loss / steps, test_loss])
            
            if test_loss < min_loss and test_loss >= 0. and (loss / steps) > 0.:
                min_loss = test_loss
                print('\t *Update min_loss')
                if not save_path is None:
                    torch.save(self.hs.state_dict(), save_path)
        
        return hloss
    
    def train_projection(self, optim, batch_size, epochs, device, hs_weights=None, save_path=None, shuffle=True, **kwargs):
        
        self.csif = CSIF(self.dim, self.upair_dim, self.post_dim, len(self.unique_upairs), len(self.unique_posts), hs_weights=hs_weights)
        
        if str(device) == 'cuda':
            self.csif.cuda()
        else:
            self.csif.cpu()
            
        self.csif.train()
        
        self.triplets = self.triplets.float().to(device)
        
        lr = kwargs['lr'] if 'lr' in kwargs else 0.01
        if optim == 'adam':
            optim = Adam(self.csif.parameters(), lr=lr)
        
        elif optim == 'adagrad':
            optim = Adagrad(self.csif.parameters(), lr=lr)
        
        elif optim == 'sgd':
            optim = SGD(self.csif.parameters(), lr=lr)
        
        steps = len(self.triplet_dict) // batch_size
        
        hloss = []
        min_loss = 1000.
        
        for epoch in range(epochs):
            print('- Epoch ', epoch)
            loss = 0.0
            for i in range(steps):
                # randomly select triplets
                indices = torch.randint(0, len(self.triplet_dict), (batch_size,)).numpy()
                batch_triplets = (self.triplets[:, indices]).to(device)
                
                _loss, _, _, _, _ = self.train_batch(i, self.csif, self.triplets, indices, optim, device, model_type='projection')
                
                loss += _loss.item()

            print('\t loss: ', loss / steps)
        
            test_loss = self.test_projection(batch_size=batch_size, device=device)
            print('\t test loss: ', test_loss)
            
            hloss.append([loss / steps, test_loss])
            
            if test_loss < min_loss and test_loss >= 0. and (loss / steps) > 0.:
                min_loss = test_loss
                print('\t *Update min_loss')
                if not save_path is None:
                    torch.save(self.csif, save_path)
                    
        return hloss
    def test_projection(self, batch_size, device, **kwargs):
        steps = len(self.test_triplet_dict) // batch_size
        batch_indices = []
        
        for i in range(steps):
            batch_indices.append(torch.randint(0, len(self.test_triplets), (batch_size,)).numpy())
        
        loss = 0.0
        
        print('Test model')
            
        self.hs.eval()
        
        for i,indices in enumerate(batch_indices):
            batch_triplets = self.test_triplets[:, indices]
            
            t_outputs = [self.test_post_dict[k] for k in indices]
            u_outputs = [self.test_upair_dict[k] for k in indices]
            
            u_indices = [self.unique_upairs_dict[u] for u in u_outputs]
            t_indices = [self.unique_posts_dict[t] for t in t_outputs]
                
            t_k = batch_triplets[-self.post_dim:].to(device)
            u_ij = batch_triplets[:self.upair_dim].to(device)
            
            _loss = self.csif(u_ij, u_indices, t_k, t_indices, device)
            
            loss += _loss.item()
            
        return loss / steps




    
   