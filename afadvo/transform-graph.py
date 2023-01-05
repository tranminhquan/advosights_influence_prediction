import argparse
import time
import pickle
import torch
from transforms.kapi import *
import warnings
# from utils.mongodb import get_af_preprocessing_db
from datetime import datetime
from bson.objectid import ObjectId
from utils.database import AFPreprocessingMongoDB

# Ignore warnings
warnings.filterwarnings("ignore")

def transform_kapi_data(database, users, posts, comments, reactions, edgelist=None, nodeattr=None, edgeattr=None, torchdata=None):
    # EDGE LIST
    print('- Creating user edge list . . .')
    start_edge_list = time.time()

    user_post_dict = group_post_by_user(df_post=posts, save_path=None)
    
    post_share = group_user_by_post(df_path=posts, df_post=posts, collection='share', save_path=None)
    post_comment = group_user_by_post(df_path=comments, df_post=posts, collection='comment', save_path=None) if not comments is None else None
    post_react = group_user_by_post(df_path=reactions, df_post=posts, collection='react', save_path=None) if not reactions is None else None

    edge_list = generate_edge_list(user_post_dict=user_post_dict, share_dict=post_share, comment_dict=post_comment, react_dict=post_react, save_path=None)
    
    if not edgelist is None:
        with open(edgelist, 'wb') as dt:
            pickle.dump(edge_list, dt)

    print('* Edge list done: ', time.time() - start_edge_list)

    # NODE ATTRIBUTE
    print('- Creating node attribute . . .')
    start_nodeattr = time.time()
    node_attr = generate_node_attribute(user_edge_list=edge_list, df_user=users, df_post=posts, user_post_dict=user_post_dict, save_path=nodeattr)
    
    print('* Node attribute done: ', time.time() - start_nodeattr)

    # EDGE ATTRIBUTE
    print('- Creating edge attribute . . .')
    start_edgeattr = time.time()
    share_count = count_by_user(df_path=posts, collection='share', user_edge_list=edge_list, user_post_dict=user_post_dict)
    comment_count = count_by_user(df_path=comments, collection='comment', user_edge_list=edge_list, user_post_dict=user_post_dict)
    react_count = count_by_user(df_path=reactions, collection='react', user_edge_list=edge_list, user_post_dict=user_post_dict)

    edge_attr = generate_edge_attribute(edge_list=[share_count, comment_count, react_count], save_path=edgeattr)
    print('* Edge list done: ', time.time() - start_edgeattr)

    # GENERATE TORCH-GEO DATA INSTANCE
    print('- Creating torch-geometric data . . .')
    start_torchgeodata = time.time()
    torch_data, node_dict = generate_data_torchgeo(edge_list=edge_list, node_atts=node_attr, edge_atts=edge_attr, reindex=True, save_path=torchdata)
    print('* Torch-geo data done: ', time.time() - start_torchgeodata)


    return edge_list, node_attr, edge_attr, torch_data, node_dict

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Tranform from raw MongoDB data to knowledge graph data')
    
    parser.add_argument('-ulist', '--uidlist', default=[], help='list of user id')
    parser.add_argument('-from', '--from_date', default=datetime(datetime.now().year, datetime.now().month - 2, datetime.now().day), help='start day to get data')
    parser.add_argument('-to', '--to_date', default=datetime.now(), help='end day to get data')

    args = parser.parse_args()

    print('- Getting data from Kapi and converting to csv . . .')
    data = export_csv(args.from_date, args.to_date, args.uidlist)

    print('- Transforming data to graph structure (this may take a while) . . .')
    edge_list, node_attr, edge_attr, torch_data, node_dict = transform_kapi_data('kapi', data[0], data[1], data[2], data[3])
    print('* Transforming process done.')

    print('- Add to database')

    # ADD TO DATABASE
    afdb = AFPreprocessingMongoDB()

    # convert nparray to conventional list
    edge_list = {k:v.tolist() for k,v in edge_list.items()}
    node_attr = {k:v.tolist() for k,v in node_attr.items()}
    edge_attr = {k:[j.tolist() for j in v] for k,v in edge_attr.items()}

    el = afdb.insert_edge_list(edge_list)
    na = afdb.insert_note_att(node_attr)
    ea = afdb.insert_edge_att(edge_attr)
    g = afdb.insert_graph_data(el.inserted_id, na.inserted_id, ea.inserted_id, torch_data, node_dict)

    # return object_id of all files
    print('Inserted to DB: ')
    print('- edge list id: ', el.inserted_id)
    print('- node att id: ', na.inserted_id)
    print('- edge att id: ', ea.inserted_id)
    print('- torch graph data id: ', g.inserted_id)

