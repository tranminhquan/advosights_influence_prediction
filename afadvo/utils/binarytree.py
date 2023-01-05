from random import shuffle
from copy import copy
import math
import numpy as np

class TreeTools:
    def __init__(self):
        #memoization for _count_nodes functions
        self._count_nodes_dict = {}
                
    def _get_subtrees(self, tree):
        yield tree
        for subtree in tree:
            if type(subtree) == list:
                for x in self._get_subtrees(subtree):
                    yield x

    # Returns pairs of paths and leafves of a tree
    def _get_leaves_paths(self, tree):
        for i, subtree in enumerate(tree):
            if type(subtree) == list:
                for path, value in self._get_leaves_paths(subtree):
                    yield [i] + path, value
            else:
                yield [i], subtree

    # Returns the number of nodes in a tree (not including root)
    def _count_nodes(self, tree):
        if id(tree) in self._count_nodes_dict:
            return self._count_nodes_dict[id(tree)]
        size = 0
        for node in tree:
            if type(node) == list:
                size += 1 + self._count_nodes(node)
        self._count_nodes_dict[id(self._count_nodes_dict)] = size
        return size


    # Returns all the nodes in a path
    def _get_nodes(self, tree, path):
        next_node = 0
        nodes = []
        for decision in path:
            nodes.append(next_node)
            next_node += 1 + self._count_nodes(tree[:decision])
            tree = tree[decision]
        return nodes
    
    
    def get_nodes_per_level(self, tree, n_leaves):
        nodes_per_level = [2]
        leak_nodes = [0]
        max_level = math.ceil(math.log(n_leaves, 2))

        # len of each path
        len_paths = np.array([len(k[0]) for k in self._get_leaves_paths(tree)])

        for level in range(1,max_level):
            leak_this_node = len(np.where(len_paths < (level+1))[0])

            leak_node = leak_nodes[-1] + (leak_this_node*2)
            leak_nodes.append(leak_node)
            nodes_per_level.append(2**(level+1) - leak_node)

        return nodes_per_level


    def get_nodes_from_path(self, path, tree, n_leaves):
    
        max_level = math.ceil(math.log(n_leaves, 2))
        indices = [0]
        node_ids = []

        for lv,p in enumerate(path):
            index = 2*indices[-1] + p
            indices.append(index)

            node_ids.append((lv,index))

        return node_ids

# turns a list to a binary tree
def random_binary_full_tree(outputs, shuffle=False):
    outputs = outputs if type(outputs) is str or np.ndarray else list(range(outputs))
    outputs = copy(outputs)
    
    if shuffle:
        shuffle(outputs)

    while len(outputs) > 2:
        temp_outputs = []
        for i in range(0, len(outputs), 2):
            if len(outputs) - (i+1) > 0:
                temp_outputs.append([outputs[i], outputs[i+1]])
            else:
                temp_outputs.append(outputs[i])
        
        outputs = temp_outputs
    return outputs