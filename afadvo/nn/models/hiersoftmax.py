import math
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam

from utils.binarytree import TreeTools, random_binary_full_tree

class HierSoftmax(nn.Module):
    
    def __init__(self, n_nodes, input_dim):
        '''
        n_nodes: #nodes
        input_dim: #dim per node
        '''
        
        super(HierSoftmax, self).__init__()
        self.n_nodes = n_nodes
        self.dim = input_dim
        self.tool = TreeTools()
        self.tree = random_binary_full_tree(outputs=range(n_nodes), shuffle=False)
        self.path_dict = {} # index: path
        
        # level of tree
        self.L = math.ceil(math.log(self.n_nodes, 2))
        
        self.indicators = torch.zeros(self.n_nodes, self.L)
        self.n_levels = torch.zeros(self.n_nodes) # level of each index
        
        # path
        for i, (path,node) in enumerate(self.tool._get_leaves_paths(self.tree)):
            ind = torch.where(torch.Tensor(path) == 0, torch.ones(len(path)), -torch.ones(len(path)))        
            self.n_levels[i] = len(ind)
            self.indicators[i] = torch.cat([ind, torch.ones(self.L - len(ind))]) if len(ind) < self.L else ind
        
        # init weights
        self.W = nn.Parameter(self.init_weights((self.n_nodes, self.dim, self.L)))
        
        
    def init_weights(self,shape):
        return torch.rand(shape, dtype=torch.float, requires_grad=True)
    
    def forward(self, inputs, label_indices, device):
        '''
        inputs shape (batch_size, dim)
        '''        
        inputs = inputs.to(device)

        self.indicators = self.indicators.to(device)

        _a = (self.indicators[label_indices, None] * self.W[label_indices]).permute(0,2,1)
        _b = inputs.unsqueeze(-1)   
        
        probs = torch.bmm(_a, _b)
        probs = probs / math.sqrt(_a.size(-1))
 
        probs = probs.squeeze(-1)
        probs = probs.sigmoid()
        probs = probs.prod(1)

        return probs, _a, _b


class JointHierSoftmax(nn.Module):
    def __init__(self, upair_input, upair_output,  post_input, post_output):
        super(JointHierSoftmax, self).__init__()
        self.hs_u = HierSoftmax(n_nodes=upair_output, input_dim=upair_input)
        self.hs_t = HierSoftmax(n_nodes=post_output, input_dim=post_input)
        
    def forward(self, upair, post, device):
        x1, _a, _b = self.hs_u(upair[0], upair[1], device)
        x2, _, _ = self.hs_t(post[0], post[1], device)

        x = -0.5 * (x1.sum().log() + x2.sum().log())
        
        return x, _a, _b
