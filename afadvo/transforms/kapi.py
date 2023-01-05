'''Transform function for Kapi DB'''

import numpy as np
import pandas as pd
import pickle
import sklearn
from sklearn import impute
import torch
from torch_geometric.data import Data
from collections import OrderedDict
# from utils.mongodb import get_kapi_db, get_af_preprocessing_db
from utils.database import AFKapiMongoDB, AFPreprocessingMongoDB

from datetime import datetime
import json

def export_csv(from_date, to_date, uid_list):
    r'''
    Read data from Kapi, then export to csv file and upload to AF preprocessing DB
    Args:
        from_date: `datetime.datetime`, begining day to get data
        to_date: `datetime.datetime`, ending day to get data
        uid_list: python list, list of ALL users, if `None`, then get all uid in kapi

    :rtype: `dictionary` of all concerned DataFrame and `InsertOneResult` that store the csv file
    '''
    nan_value = float("NaN")
    kapidb = AFKapiMongoDB()
    
    data = {}
    
    # filter_users
    print('* filter users')
    user_path = './temp_filter_user.json'
    cursor = kapidb.get_users(uid_list=uid_list)
    df_user = pd.DataFrame(list(cursor))
    temp_df_user = df_user
    df_user.replace('', np.nan, inplace=True)
    df_user.dropna(subset = ['fid'], inplace=True)
    
    print(list(df_user['fid']))
#     df.to_json(user_path, default_handler=str)
#     user_json = json.load(open(user_path))

    # posts
    print('* posts')
    post_path = './temp_posts.json'
    cursor = kapidb.get_posts(from_date, to_date)
    df_post = pd.DataFrame(list(cursor))
    post_ids = list(df_post['fid'])
    df_post.replace('', np.nan, inplace=True)
    df_post.dropna(subset = ['fid', 'from_user'], inplace=True)
#     df.to_json(post_path, default_handler=str)
#     post_json = json.load(open(post_path))
    
    # comments
    print('* comments')
    cmt_path = './temp_comments.json'
    cursor = kapidb.get_comments(from_date, to_date)
    df_comment = pd.DataFrame(list(cursor))
    df_comment.replace('', np.nan, inplace=True)
    df_comment.dropna(subset = ['fid', 'from_user'], inplace=True)
#     df.to_json(cmt_path, default_handler=str)
#     comment_json = json.load(open(cmt_path))
    
    # reactions
    print('* reactions')
    react_path = './temp_reactions.json'
    cursor = kapidb.get_reactions(post_ids)
    
#     df = pd.DataFrame()
        
    df_react = pd.DataFrame(list(cursor), columns=['fid', 'reaction_type', 'from_user_id', 'to_user_id', 'content_type'])
    df_react.replace('', np.nan, inplace=True)
    df_react.dropna(subset = ['fid', 'from_user_id'], inplace=True)
    
#     df.to_json(react_path, default_handler=str)
#     react_json = json.load(open(react_path))

#     afdb = AFPreprocessingMongoDB()
#     oid = afdb.insert_af_files(user_json, post_json, comment_json, react_json)
        
    return df_user, df_post, df_comment, df_react, temp_df_user

def group_post_by_user(df_post='../files/chosen_post_2.csv', save_path=None):
    r'''
    Read post.csv collection and return dictionary of user_id - post_ids
    
    Args:
        df_post: str or Pandas Dataframe. If str, df_post should be path to csv file
        save_path: path to save. If None, save process is skipped
    
    :rtype: python dictionary (user_post_dict: dictionary of user_id - post_ids)
    '''
    
    if type(df_post) is str:
        df_post = pd.read_csv(df_post, dtype={'fid': str, 'from_user': str, 'to_user': str, 'parent_id': str})
    else:
        df_post = df_post.astype({'fid': str, 'from_user': str, 'to_user': str, 'parent_id': str})
        
    user_post_dict = dict(df_post.drop(df_post.columns.difference(['from_user', 'fid']), 1, inplace=False).dropna().drop_duplicates().groupby('from_user')['fid'].apply(list))
    
    if not save_path is None:
        with open(save_path, 'wb') as dt:
            pickle.dump(user_post_dict, dt)
    
    return user_post_dict

def group_user_by_post(df_path, df_post, collection, save_path=None):
    r'''
    Read df.csv collection and return dictionary of post_id - user_ids
    
    Args:
        df_path: str. Path to DataFrame need to query (share, comment or react), df_path should be path to csv file
        df_post: str or Pandas Dataframe. DataFrame of chosen post. If str, df should be path to csv file
        collection: str, collection should be one of 'share', 'comment' or 'reaction'
        save_path: path to save. If None, save process is skipped

    :rtype: python dictionary
    '''
    
    if type(df_post) is str:
        df_post = pd.read_csv(df_post, dtype={'fid': str, 'from_user': str, 'to_user': str, 'parent_id': str})
    else:
        df_post = df_post.astype({'fid': str, 'from_user': str, 'to_user': str, 'parent_id': str})
        
    if collection == 'share':

        df = df_post.dropna(subset=['parent_id', 'from_user'])
        post_user_dict = dict(df.groupby('fid')['from_user'].apply(list))
        
    # Due to over size of comment and react, these csv file should be read in chunk
    elif collection == 'comment':
        # get all user ids who comments on those posts
        post_ids = list(df_post.fid)
        
        if type(df_path) is str:
            chunksize = 1e7
            cmt_user_ids = pd.DataFrame()
            for i,df in enumerate(pd.read_csv(df_path, dtype={'fid': str, 'post_id':str, 'from_user': str, 'to_user': str}, chunksize=chunksize)):
                df_temp = df[df.post_id.isin(post_ids)]
                cmt_user_ids = cmt_user_ids.append(df_temp, ignore_index=True)
        else:
            df_path = df_path.astype({'fid': str, 'post_id':str, 'from_user': str, 'to_user': str})
            cmt_user_ids = pd.DataFrame(df_path[df_path.post_id.isin(post_ids)])
            
        post_user_dict = dict(cmt_user_ids.drop(cmt_user_ids.columns.difference(['from_user','post_id']), 1, inplace=False).dropna().drop_duplicates().groupby('post_id')['from_user'].apply(list))
        
    elif collection == 'react' or collection == 'reaction':
        # get all user ids who comments on those posts
        post_ids = list(df_post.fid)
        
        if type(df_path) is str:
            chunksize = 1e7 
            react_user_ids = pd.DataFrame()
            for i,df in enumerate(pd.read_csv(df_path, dtype={'fid': str, 'from_user_id': str}, chunksize=chunksize)):
                df_temp = df[df.fid.isin(post_ids)]
                react_user_ids = react_user_ids.append(df_temp, ignore_index=True)
        else:
            df_path = df_path.astype({'fid': str, 'from_user_id': str})
            react_user_ids = pd.DataFrame(df_path[df_path.fid.isin(post_ids)])
        
        react_user_ids = react_user_ids.drop(react_user_ids.columns.difference(['fid','from_user_id']), 1, inplace=False).dropna().drop_duplicates()
        post_user_dict = dict(react_user_ids.dropna().drop_duplicates().groupby('fid')['from_user_id'].apply(list))
     
    if not save_path is None:
        with open(save_path, 'wb') as dt:
            pickle.dump(post_user_dict, dt)
            
    return post_user_dict

def generate_edge_list(user_post_dict='../files/group_post_by_user_in_post.hdf5', 
                       share_dict='../files/group_user_by_post_in_share.hdf5', 
                       comment_dict='../files/group_user_by_post_in_comment.hdf5', 
                       react_dict='../files/group_user_by_post_in_reaction.hdf5', 
                       save_path=None):
    
    r'''
    Generate edge list dictionary from dedicated dictionary of user_id - post_ids, and post_id - user_ids who share, comment and react
    
    Args:
        user_post_dict: str or python dict, user_id - post_ids dict. If str, it should be available path
        share_dict : user_id - post_ids who share
        comment_dict: user_id - post_ids who comment
        react_dict: user_id - post_ids who react
        save_path: str. If None, save process is skipped
    
    :rtype: python dictionary (user_edge_list)
    '''
    
    # load file
    
    if type(share_dict) is str:
        with open(share_dict, 'rb') as dt:
            share_dict = pickle.load(dt)

    if type(user_post_dict) is str:
        with open(user_post_dict, 'rb') as dt:
            user_post_dict = pickle.load(dt)

    if type(comment_dict) is str:
        with open(comment_dict, 'rb') as dt:
            comment_dict = pickle.load(dt)

    if type(react_dict) is str:
        with open(react_dict, 'rb') as dt:
            react_dict = pickle.load(dt)
            
    print('- user_post_dict records: ', len(user_post_dict))
    print('- share_dict records: ', len(share_dict))
    print('- comment_dict records: ', len(comment_dict))
    print('- react_dict records: ', len(react_dict))
    
            
    user_edge_list = {}
    counter = 0
    n = 0
    for k,v in user_post_dict.items():

        for pid in v:
            # share
            if pid in share_dict:
                # user id is first part of '_' in parent_id
                _share = [str(k) for k in share_dict[pid]]
            else:
                _share = []

            # comment
            if pid in comment_dict:
                _comment = [str(k) for k in comment_dict[pid]]
            else:
                _comment = []

            # reaction
            if pid in react_dict:
                _react = [str(k) for k in react_dict[pid]]
            else:
                _react = []

            # concat list
            n += 1
            if len(_share) == 0 and len(_comment) == 0 and len(_react) == 0:
                counter += 1
            else:
                uids = np.unique(np.array(_share + _comment + _react, dtype=str))
                idx = np.where(uids == str(k))[0]
                
                if len(idx) > 0:
                    uids = np.delete(uids, idx)
                    
                    if len(uids) == 0:
                        continue
                
                user_edge_list[str(k)] = uids
                
    print('- Number of isolated nodes: ', counter, ' | Percentage: ', (counter/n)*100)
    
    return user_edge_list

def count_by_user(df_path, collection,
                  user_edge_list='../files/user_edge_list_3.hdf5', 
                  user_post_dict='../files/group_post_by_user_in_post.hdf5',
                  chunksize=1e7,
                  save_path=None):
    r'''
    Counter number of shares, comments or reactions by user
    
    Args:
        df_path: path to concerned Pandas Dataframe
        user_edge_list: str or DataFrame. if str, it should be available path
        user_post_dict: str or dictionary
        collections: str. one of 'share', 'comment' or 'react'
    
    :rtype: python dictionary, (number of shares/comments/reacts by user_id)
    '''
    
    if type(user_edge_list) is str:
        with open(user_edge_list, 'rb') as dt:
            user_edge_list = pickle.load(dt)
    
    if type(user_post_dict) is str:
        with open(user_post_dict, 'rb') as dt:
            user_post_dict = pickle.load(dt)
    
    edge_count = {}
    n_counter = 0
    
    if collection == 'share':
        edge_count = count_by_user_share(df_path, user_edge_list, user_post_dict, chunksize)
    elif collection == 'comment':
        edge_count = count_by_user_comment(df_path, user_edge_list, user_post_dict, chunksize)
    elif collection == 'react' or collection == 'reaction':
        edge_count = count_by_user_react(df_path, user_edge_list, user_post_dict, chunksize)

    if not save_path is None:
        with open(save_path, 'wb') as dt:
            pickle.dump(edge_count, dt)
            
    return edge_count
                          

def count_by_user_share(df_path,
                  user_edge_list='../files/user_edge_list_3.hdf5', 
                  user_post_dict='../files/group_post_by_user_in_post.hdf5',
                  chunksize=10000):
    if type(user_edge_list) is str:
        with open(user_edge_list, 'rb') as dt:
            user_edge_list = pickle.load(dt)
    
    if type(user_post_dict) is str:
        with open(user_post_dict, 'rb') as dt:
            user_post_dict = pickle.load(dt)
    
    edge_count = {}
    
    if type(df_path) is str:
        for i,df in enumerate(pd.read_csv(df_path, dtype = {'fid':str, 'from_user':str, 'parent_id': str}, chunksize=chunksize)):
            for k,v in user_edge_list.items():

                pids = user_post_dict[str(k)]
                _dict = dict(df[df.from_user.isin(v) & df.parent_id.isin(pids)].groupby('from_user')['fid'].count())

                if not k in edge_count:
                    edge_count[k] = np.zeros(len(v), dtype=int)
                else:
                    n_counts = edge_count[k]
                    for u,j in _dict.items():
                        idx = np.where(v == str(u))[0][0]
                        n_counts[idx] = j

                    edge_count[k] = n_counts
    else:
        df_path = df_path.astype({'fid':str, 'from_user':str, 'parent_id': str})
        for k,v in user_edge_list.items():
        
            pids = user_post_dict[str(k)]
            _dict = dict(df_path[df_path.from_user.isin(v) & df_path.parent_id.isin(pids)].groupby('from_user')['fid'].count())
            
            n_counts = np.zeros(len(v))
            
            for u,j in _dict.items():
                idx = np.where(v == str(u))[0][0]
                n_counts[idx] = j
            
            edge_count[k] = n_counts
            
        
#         df_path = df_path.astype({'fid':str, 'from_user':str, 'parent_id': str})
#         for k,v in user_edge_list.items():

#             pids = user_post_dict[str(k)]
#             _dict = dict(df_path[df_path.from_user.isin(v) & df_path.parent_id.isin(pids)].groupby('from_user')['fid'].count())

#             if not k in edge_count:
#                 edge_count[k] = np.zeros(len(v), dtype=int)
#             else:
#                 n_counts = edge_count[k]
#                 for u,j in _dict.items():
#                     idx = np.where(v == str(u))[0][0]
#                     n_counts[idx] = j

#                 edge_count[k] = n_counts
                          
    return edge_count
                          
def count_by_user_comment(df_path,
                  user_edge_list='../files/user_edge_list_3.hdf5', 
                  user_post_dict='../files/group_post_by_user_in_post.hdf5',
                  chunksize=10000):
    if type(user_edge_list) is str:
        with open(user_edge_list, 'rb') as dt:
            user_edge_list = pickle.load(dt)
    
    if type(user_post_dict) is str:
        with open(user_post_dict, 'rb') as dt:
            user_post_dict = pickle.load(dt)
    
    edge_count = {}
    
    if type(df_path) is str:
        for i,df in enumerate(pd.read_csv(df_path, dtype = {'fid': str, 'post_id':str, 'from_user': str, 'to_user': str}, 
                                          chunksize=chunksize)):
            for k,v in user_edge_list.items():

                pids = user_post_dict[str(k)]
                _dict = dict(df[df.from_user.isin(v) & df.post_id.isin(pids)].groupby('from_user')['post_id'].count())

                if not k in edge_count:
                    edge_count[k] = np.zeros(len(v), dtype=int)
                else:
                    n_counts = edge_count[k]
                    for u,j in _dict.items():
                        idx = np.where(v == str(u))[0][0]
                        n_counts[idx] = j

                    edge_count[k] = n_counts
    else:
        df_path = df_path.astype({'fid': str, 'post_id':str, 'from_user': str, 'to_user': str})
        for k,v in user_edge_list.items():
        
            pids = user_post_dict[str(k)]
            _dict = dict(df_path[df_path.from_user.isin(v) & df_path.post_id.isin(pids)].groupby('from_user')['post_id'].count())
            
            n_counts = np.zeros(len(v))
            
            for u,j in _dict.items():
                idx = np.where(v == str(u))[0][0]
                n_counts[idx] = j
            
            edge_count[k] = n_counts
        
        
#         df_path = df_path.astype({'fid': str, 'post_id':str, 'from_user': str, 'to_user': str})
#         for k,v in user_edge_list.items():
                          
#             pids = user_post_dict[str(k)]
#             _dict = dict(df_path[df_path.from_user.isin(v) & df_path.post_id.isin(pids)].groupby('from_user')['post_id'].count())

#             if not k in edge_count:
#                 edge_count[k] = np.zeros(len(v), dtype=int)
#             else:
#                 n_counts = edge_count[k]
#                 for u,j in _dict.items():
#                     idx = np.where(v == str(u))[0][0]
#                     n_counts[idx] = j

#                 edge_count[k] = n_counts
                          
    return edge_count
                          
def count_by_user_react(df_path,
                  user_edge_list='../files/user_edge_list_3.hdf5', 
                  user_post_dict='../files/group_post_by_user_in_post.hdf5',
                  chunksize=1e7):
    if type(user_edge_list) is str:
        with open(user_edge_list, 'rb') as dt:
            user_edge_list = pickle.load(dt)
    
    if type(user_post_dict) is str:
        with open(user_post_dict, 'rb') as dt:
            user_post_dict = pickle.load(dt)
    
    edge_count = {}
                   
    if type(df_path) is str:
        for i,df in enumerate(pd.read_csv(df_path, dtype = {'fid': str, 'from_user_id': str}, 
                                          chunksize=chunksize)):
            for k,v in user_edge_list.items():

                pids = user_post_dict[str(k)]
                _dict = dict(df[df.from_user_id.isin(v) & df.fid.isin(pids)].groupby('from_user_id')['fid'].count())

                if not k in edge_count:
                    edge_count[k] = np.zeros(len(v), dtype=int)
                else:
                    n_counts = edge_count[k]
                    for u,j in _dict.items():
                        idx = np.where(v == str(u))[0][0]
                        n_counts[idx] = j

                    edge_count[k] = n_counts
    else:
        df_path = df_path.astype({'fid': str, 'from_user_id': str})
        for k,v in user_edge_list.items():
        
            pids = user_post_dict[str(k)]
            _dict = dict(df_path[df_path.from_user_id.isin(v) & df_path.fid.isin(pids)].groupby('from_user_id')['fid'].count())
            
            n_counts = np.zeros(len(v))
            
            for u,j in _dict.items():
                idx = np.where(v == str(u))[0][0]
                n_counts[idx] = j
            
            edge_count[k] = n_counts
        
        
#         df_path = df_path.astype({'fid': str, 'from_user_id': str})
#         for k,v in user_edge_list.items():
                          
#             pids = user_post_dict[str(k)]
#             _dict = dict(df_path[df_path.from_user_id.isin(v) & df_path.fid.isin(pids)].groupby('from_user_id')['fid'].count())

#             if not k in edge_count:
#                 edge_count[k] = np.zeros(len(v), dtype=int)
#             else:
#                 n_counts = edge_count[k]
#                 for u,j in _dict.items():
#                     idx = np.where(v == str(u))[0][0]
#                     n_counts[idx] = j

#                 edge_count[k] = n_counts
                          
    return edge_count


def generate_edge_attribute(edge_list = ['../files/edge_share_3.hdf5', 
                                         '../files/edge_comment_3.hdf5', 
                                         '../files/edge_react_3.hdf5'], 
                            save_path=None):
    r'''
    Generate edge attribute from list of edges
    
    Args:
        edge_list: [str] or [dictionary]. List of edge attribute count
        save_path: str or None.

    :rtype: python dictionary (feature edge dictionary)
    '''
    
    if not type(edge_list) is list or not type(edge_list) is np.ndarray:
        edge_list = np.asarray(edge_list)
        
    # print(edge_list)
    
    if type(edge_list[0]) is str:
        edge_attrs = []
        for path in edge_list:
            with open(path, 'rb') as dt:
                temp = pickle.load(dt)
                edge_attrs.append(temp)
    else:
        edge_attrs = edge_list


    edge_feature = {}
    
    for k,v in edge_attrs[0].items():
        atts = []
        for i in range(len(v)):
            _att = np.asarray([d[k][i] for d in edge_attrs], dtype=int)
            atts.append(_att)
        
        edge_feature[k] = atts
    
    if not save_path is None:
        with open(save_path, 'wb') as dt:
            pickle.dump(edge_feature, dt)
    
    return edge_feature

def generate_node_attribute(user_edge_list='../files/user_edge_list_3.hdf5', 
                           df_user='../files/chosen_users.csv',
                            df_post=None, user_post_dict=None,
                           attrs=['total_follower', 'total_friend', 'books_count', 'films_count', 'music_count', 'restaurants_count', 'sex', 'active', 'post_fame'], 
                           save_path=None):
    r'''
    Generate node attributes by given list
    
    Args:
        user_edge_list: str or dict python. If str, it should be path
        df_user: str or DataFrame, user dataframe
        attrs: list, name of concerned attributes from df_user
        save_path: str or None. If None, save process is skipped

    :rtype: python dictionary (node_atts_list: dict, key is node_id, values is array of concerned attributes)
    '''
    
    if type(user_edge_list) is str:
        with open(user_edge_list, 'rb') as dt:
            user_edge_list = pickle.load(dt)
            
    if type(df_user) is str:
        df_user = pd.read_csv(df_user, dtype={'fid': str})
    else:
        df_user = df_user.astype({'fid': str})
        
    if not df_post is None:
        df_post = df_post.astype({'fid': str, 'from_user': str, 'to_user': str})
        
    # get all distinct uids
    unique_uid_keys = np.unique(list(user_edge_list.keys()))
    unique_uid_values = np.unique(np.concatenate(list(user_edge_list.values())))
    uids = np.unique(np.concatenate([unique_uid_keys, unique_uid_values]))

    # filtered in df_user
    df_filtered_user = df_user[df_user.fid.isin(uids)]
    
    
    
    # cal active and post_fame
    if not 'active' in df_filtered_user.columns.tolist():
        post_count = {k: len(v) for k,v in user_post_dict.items()}
        
        active_dict = get_frequency_by_user(df_post, post_count)
        df_filtered_user['active'] = df_filtered_user['fid'].map(active_dict)
    
    if not 'post_fame' in df_filtered_user.columns.tolist():
        fame_dict = get_post_fame(user_edge_list, df_post)
        df_filtered_user['post_fame'] = df_filtered_user['fid'].map(fame_dict)
        
    df_filtered_user.replace(np.nan, 0.0, inplace=True)

    if 'sex' in attrs:
        df_filtered_user.sex = df_filtered_user.sex.map({'Nam': 1.0, 'Nữ': 2.0, 'Gay': 3.0, 'Mộc Nhiên': 2.0, 'Đàn Ông (Tính Thì Đàn Bà)': 1.0})
        df_filtered_user.sex = df_filtered_user.sex.fillna(0.0)
        

    # fillna series
#     for att in attrs:
#         # print(df_filtered_user[[att]])
#         impt = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
#         impt = impt.fit(df_filtered_user[[att]])
#         df_filtered_user[att] = impt.transform(df_filtered_user[[att]])
    
    attrs += ['fid']

#     df_filtered_user = df_filtered_user.fillna(value=fill_data).drop(df_filtered_user.columns.difference(attrs), 1, inplace=False)
    df_filtered_user = df_filtered_user.drop(df_filtered_user.columns.difference(attrs), 1, inplace=False)
        
    # create node_atts_list dictionary
    values_att = df_filtered_user.drop('fid', axis=1).values
    key_att = list(df_filtered_user['fid'])
    node_atts = dict(zip(key_att, values_att))
    
    if not save_path is None:
        with open(save_path, 'wb') as dt:
            pickle.dump(node_atts, dt)

    print('*TEMPORARY! Add uids that are not in df_user, all attributes are .0 deafaults')
    for uid in uids:
        if uid not in key_att:
            node_atts[str(uid)] = np.zeros(len(attrs) - 1, dtype='float')
    
    return node_atts
    
    return node_atts


def get_frequency_by_user(df_post='/tf/data/adv/node_embedding/chosen_post_2.csv', 
                          post_count='/tf/data/adv/node_embedding/count_post_by_user.hdf5'):

    r'''
    Get the active day of users.
        active day = range of min - max day - days having posts

    Args:
        df_post: dataframe contains posts data
        post_count: python dictionary, post counting by user
    
    :rtype: python dictionary

    '''
    
    if type(df_post) is str:
        df_post = pd.read_csv(df_post, dtype={'fid': str, 'from_user': str, 'to_user': str})
    else:
        df_post = df_post.astype({'fid': str, 'from_user': str, 'to_user': str})
        
    if type(post_count) is str:
        with open(post_count, 'rb') as dt:
            post_count = pickle.load(dt)

    df_post['created_date'] = df_post.created_date.astype('datetime64[ns]')
    
    active_dict = dict((df_post.groupby('from_user')['created_date'].max() - df_post.groupby('from_user')['created_date'].min()).dt.days)
    
    for k,v in active_dict.items():
        active_dict[k] = float(v) / post_count[str(k)]
    
    return active_dict


def get_post_fame(user_edge_list, df_post):
    r'''
    Calculating fame of posts by users
    fame of post per user = (0.1*reactions + 0.3*comments + 0.6*shares) / number of posts per
    
    Args:
        user_edge_list: user edge list hdf5 file
        df_post: str or dataframe
        
    :rtype: python dictionary
    '''
    
    if type(user_edge_list) is str:
        with open(user_edge_list, 'rb') as dt:
            user_edge_list = pickle.load(dt)
            
    if type(df_post) is str:
        df_post = pd.read_csv(df_post, dtype={'fid':str, 'from_user': str, 'likes_count': float, 'comments_count': float, 'shares_count': float})
    else:
        df_post = df_post.astype({'fid':str, 'from_user': str, 'likes_count': float, 'comments_count': float, 'shares_count': float})
    
    unique_uid_keys = np.unique(list(user_edge_list.keys()))
    unique_uid_values = np.unique(np.concatenate(list(user_edge_list.values())))
    uids = np.unique(np.concatenate([unique_uid_keys, unique_uid_values]))
    
    dicts = dict(df_post[df_post.from_user.isin(uids)].groupby('from_user')['likes_count', 'comments_count', 'shares_count'].count().apply(dict))
    
    
    return {k: (0.1*dicts['likes_count'][k]) + (0.3*dicts['comments_count'][k]) + (0.6*dicts['shares_count'][k]) for k,_ in dicts['likes_count'].items()}


def generate_data_torchgeo(edge_list='/tf/data/adv/node_embedding/release_v20/graph_data/user_edge_list_3.hdf5', 
                           node_atts='/tf/data/adv/node_embedding/release_v20/graph_data/node_atts_3.hdf5', 
                           edge_atts='/tf/data/adv/node_embedding/release_v20/graph_data/edge_feature_3.hdf5', 
                           reindex=True, save_path=None):
    '''
    Generate torch-geometric Data type
    
    Args:
        edge_list: python dict or str.
        node_atts: python dict or str.
        edge_atts: python dict or str.
        reindex: bool. If True, re-index all nodes start from 0
        save_path
    ---
    :rtype:
        :class:`torch_geometric.data.Data`
        python dictionary (re-indexing)
    '''
    
    if type(edge_list) is str:
        with open(edge_list, 'rb') as dt:
            edge_list = pickle.load(dt)
            
    if type(node_atts) is str:
        with open(node_atts, 'rb') as dt:
            node_atts = pickle.load(dt)
            
    if type(edge_atts) is str:
        with open(edge_atts, 'rb') as dt:
            edge_atts = pickle.load(dt)

    # ordering dictg
    edge_list = OrderedDict(sorted(edge_list.items()))
    node_atts = OrderedDict(sorted(node_atts.items()))
    edge_atts = OrderedDict(sorted(edge_atts.items()))
            
        
    # calculate number of edges
    n_edge = 0
    for k,v in edge_list.items():
        n_edge += len(v)
        
    
    # --------create edge_indicies to fit torch-geo data----------
    # print('- Create edge indices to fit torch-geo data from edge list')
    edge_indices = []
    
    for k,v in edge_list.items():
        for j in v:
            edge_indices.append([str(k), str(j)])
    
    edge_indices = np.asarray(edge_indices).T
    
    if reindex is True:
        node_dict = dict(zip(np.unique(edge_indices).tolist(), torch.arange(len(np.unique(edge_indices))).tolist()))
        
        t_edge_indices = torch.zeros((2, n_edge), dtype=torch.long)
        for i, r in enumerate(edge_indices):
            for j, c in enumerate(r):
                t_edge_indices[i,j] = node_dict[str(c)]
    else:
        node_dict = None
        t_edge_indices = edge_indices
        
    # -----create feature matrix to fit torch-geo data, convert to pytorch structe [num_nodes, num_node_features]-----
    if not node_atts is None:
        n_node_atts = len(list(node_atts.values())[0])
    else:
        n_node_atts = 0
    
    feature_matrix = torch.zeros((len(node_atts), n_node_atts), dtype=torch.float)
    print('feature matrix: ', feature_matrix.shape)
    for i, (k,v) in enumerate(node_atts.items()):
        feature_matrix[i, :] = torch.from_numpy(v)

    # ----- create edge attributes to fit torch-geo data----------
    if not edge_atts is None:
        n_edge_atts = len(list(edge_atts.values())[0][0])
    else:
        n_edge_atts = 0
        
    edge_features = torch.zeros((n_edge, n_edge_atts), dtype=torch.float)
    print('edge_features: ', edge_features.shape)
    counter = 0
    for i, (k,v) in enumerate(edge_atts.items()):
        for j in range(len(v)):
            edge_features[counter,:] = torch.from_numpy(v[j])
            counter += 1
    
#     for i, (k,v) in enumerate(edge_atts.items()):
#         edge_features[i, :] = torch.from_numpy(v[0])
        
    data = Data(x=feature_matrix, edge_index=t_edge_indices, edge_attr=edge_features)
    
    if not save_path is None:
        torch.save(data, save_path)
    
    return data, node_dict