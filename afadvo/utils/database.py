import abc
import pymongo
from pymongo import MongoClient
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
from bson.objectid import ObjectId
import os

from config.database import kapi
from config.database import preprocessing

class AFKapiDBInterface(metaclass=abc.ABCMeta):
    def __init__(self, host, username, password, authSource, authMechanism):
        self.client = MongoClient(host, username=username, 
                                password=password, authSource=authSource, 
                                authMechanism=authMechanism)
        self.db = self.client[authSource]

    @abc.abstractmethod
    def get_users(self, uid_list):
        raise NotImplementedError

    @abc.abstractmethod
    def get_posts(self, from_date, to_date, uidlist):
        raise NotImplementedError

    @abc.abstractmethod
    def get_comments(self, from_date, to_date, post_ids):
        raise NotImplementedError

    @abc.abstractmethod
    def get_reactions(self, post_ids):
        raise NotImplementedError


class AFKapiMongoDB(AFKapiDBInterface):
    def __init__(self, host=kapi['mongo']['host'], username=kapi['mongo']['username'], 
                password=kapi['mongo']['password'], authSource=kapi['mongo']['authSource'], 
                authMechanism=kapi['mongo']['authMechanism']):
        super(AFKapiMongoDB, self).__init__(host, username, password, authSource, authMechanism)
    
    def get_users(self, uid_list=[]):
        user_cursor = self.db['filter_users'].find({'fid':{'$in': uid_list}}) if len(uid_list) > 0 else self.db['filter_users'].find()
        return user_cursor

    def get_posts(self, from_date, to_date, uidlist=None):
        date_query = {}
        
        if uidlist is None:
            date_query['$and'] = [{'created_date':{'$gte': from_date}}, {'created_date':{'$lte': to_date}}]
        else:
            date_query['$and'] = [{'from_user':{'$in': uidlist}}, {'created_date':{'$gte': from_date}}, {'created_date':{'$lte': to_date}}]
            
        cursor = self.db['posts'].find(date_query)
        
        return cursor

    def get_comments(self, from_date, to_date, post_ids=None):
        date_query = {}
        if post_ids is None:
            date_query['$and'] = [{'created_date':{'$gte': from_date}}, {'created_date':{'$lte': to_date}}]
        else:
            date_query['$and'] = [{'post_id':{'$in': post_ids}}, {'created_date':{'$gte': from_date}}, {'created_date':{'$lte': to_date}}]
            
        cursor = self.db['comments'].find(date_query)
        return cursor


    def get_reactions(self, post_ids):
        cursor = self.db['reactions'].find({'fid':{'$in': post_ids}}, batch_size=1000)
        return cursor


#########################################################################################################################################################################

class AFPreprocessingDBInterface(metaclass=abc.ABCMeta):
    def __init__(self, host, username, password, authSource, authMechanism='SCRAM-SHA-1'):
        self.client = MongoClient(host, username=username, password=password, authSource=authSource, authMechanism=authMechanism)
        self.db = self.client[authSource]

    @abc.abstractmethod
    def insert_af_files(self, user_json, post_json, comment_json, react_json, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def insert_edge_list(self, edge_list, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def insert_note_att(self, node_att, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def insert_edge_att(self, edge_att, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def insert_graph_data(self, edge_list_id, node_att_id, edge_att_id, torch_data, node_dict, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def insert_model_info(self, model_info, data_id, history, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_graph_data(self, data_id, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_model_info(self, model_id, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def insert_embedding(self, embedding_path, model_id, data_id, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_embedding(self, embedding_id):
        raise NotImplementedError

class AFPreprocessingMongoDB(AFPreprocessingDBInterface):
    def __init__(self, host=preprocessing['mongo']['host'], username=preprocessing['mongo']['username'], 
                password=preprocessing['mongo']['password'], authSource=preprocessing['mongo']['authSource'], 
                authMechanism=preprocessing['mongo']['authMechanism']):
        super(AFPreprocessingMongoDB, self).__init__(host, username, password, authSource, authMechanism)

    def insert_af_files(self, user_json, post_json, comment_json, react_json, **kwargs):
        data = {}
        data['date'] = datetime.now()
        data['filter_users'] = user_json
        data['posts'] = post_json
        data['comments'] = comment_json
        data['reactions'] = react_json

        _id = self.db['AF_files'].insert_one(data)

        return _id

    def insert_edge_list(self, edge_list, **kwargs):
        edge_list_data = {}
        edge_list_data['date'] = datetime.now()
        edge_list_data['data'] = pickle.dumps(edge_list)
        return self.db['AF_edge_list'].insert_one(edge_list_data)

    def insert_note_att(self, node_attr, **kwargs):
        node_attr_data = {}
        node_attr_data['date'] = datetime.now() if not 'date' in kwargs else kwargs['date']
        node_attr_data['data'] = pickle.dumps(node_attr)
        return self.db['AF_node_att'].insert_one(node_attr_data)

    def insert_edge_att(self, edge_attr, **kwargs):
        edge_attr_data = {}
        edge_attr_data['date'] = datetime.now() if not 'date' in kwargs else kwargs['date']
        edge_attr_data['data'] = pickle.dumps(edge_attr)
        return self.db['AF_edge_att'].insert_one(edge_attr_data)

    def insert_graph_data(self, edge_list_id, node_att_id, edge_att_id, torch_data, node_dict, **kwargs):
        '''
        use pickle.dumps for `torch_data` and `node_dict`
        '''
        torchgraph_data = {}
        torchgraph_data['date'] = datetime.now() if not 'date' in kwargs else kwargs['date']
        torchgraph_data['edge_list_id'] = edge_list_id
        torchgraph_data['node_att_id'] = edge_list_id
        torchgraph_data['edge_att_id'] = edge_list_id
        torchgraph_data['data'] = pickle.dumps(torch_data)
        torchgraph_data['node_dict'] = pickle.dumps(node_dict)
        return self.db['AF_graph_data'].insert_one(torchgraph_data)

    def insert_model_info(self, model_info, data_id, history, **kwargs):
        model_data = {}
        model_data['date'] = datetime.now() if not 'date' in kwargs else kwargs['date']
        model_data['data_id'] = ObjectId(str(data_id))
        model_data['model_info'] = pickle.dumps(model_info)
        model_data['history'] = pickle.dumps(history)

        return self.db['AF_model'].insert_one(model_data)

    def get_graph_data(self, data_id, **kwargs):
        return self.db['AF_graph_data'].find_one({'_id': ObjectId(str(data_id))})

    def get_model_info(self, model_id, **kwargs):
        return self.db['AF_model'].find_one({'_id': ObjectId(str(model_id))})

    def insert_embedding(self, embedding_path, model_id, data_id, **kwargs):
        embedding_data = {}
        embedding_data['date'] = datetime.now() if not 'date' in kwargs else kwargs['date']
        embedding_data['data_id'] = data_id
        embedding_data['model_id'] = model_id
        embedding_data['path'] = embedding_path

        return self.db['AF_embedding'].insert_one(embedding_data)

    def get_embedding(self, embedding_id):
        # from AF_embedding
        embdt = self.db['AF_embedding'].find_one({'_id': ObjectId(str(embedding_id))})
        emb_path = embdt['path']
        emb_path = os.path.join(os.path.dirname(os.getcwd()), emb_path)

        data_id = embdt['data_id']

        # from AF_graph_data
        graphdt = self.db['AF_graph_data'].find_one({'_id': ObjectId(str(data_id))})
        node_dict = pickle.loads(graphdt['node_dict'])
        edge_list_id = graphdt['edge_list_id']

        # from AF_edge_list
        eldt = self.db['AF_edge_list'].find_one({'_id': ObjectId(str(edge_list_id))})
        edge_list = pickle.loads(eldt['data'])

        return emb_path, node_dict, edge_list
        
