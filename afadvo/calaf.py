'''Calculation AF score'''

import argparse
from utils.predict import get_similarity_2, cal_af_2 
import pickle
import warnings
from config.calaf import *
from utils.database import AFPreprocessingMongoDB

# Ignore warnings
warnings.filterwarnings("ignore")


def cal_af(uid, embedding_id, threshold, limit, device):
    
    # load embedding from AF_embedding collection
    afdb = AFPreprocessingMongoDB()
    embedding_path, node_dict, edge_list = afdb.get_embedding(embedding_id)

    dirs, probs, degs = get_similarity_2(uid, edge_list, embedding_path, node_dict, threshold, limit)
    af_score = cal_af_2(uid, dirs, probs, degs)

    # TO-DO: insert to AF ready-to-use DB
    #

    return af_score, dirs, probs, degs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate AF from given node embedding')
    parser.add_argument('-uid', '--uid', help='user id')
    parser.add_argument('-embid', '--embedding_id', help='embedding vector or can be path to file')
    parser.add_argument('-gpu', '--gpu', help='acclerate by gpu')
    args = parser.parse_args()

    device = 'gpu' if args.gpu is True else 'cpu'
    
    # temporary: force to set cpu
    device = 'cpu'

    af_score, dirs, probs, degs = cal_af(uid=args.uid, embedding_id=args.embedding_id, threshold=calaf['THRESHOLD'], 
            limit=calaf['LEVELS'], device=device)

    print('* af score: ', af_score)
    print('* dirs: ', dirs)
    print('* probs: ', probs)
    print('* degs: ', degs)

    
