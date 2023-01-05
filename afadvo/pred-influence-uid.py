'''Calculation AF score'''

import argparse
from utils.pred_post_inf import predict
import pickle
import warnings
from utils.pred_post_inf import PostInfluencePredict

# Ignore warnings
warnings.filterwarnings("ignore")


def pred_influence_to_uid(from_uid, to_uid, post_content, config=None):
    
    # load embedding from AF_embedding collection
    post_predictor = PostInfluencePredict(config)

    prob = post_predictor.predict(from_uid, to_uid, post_content)

    return prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict influence probability given a content of a user to another specific user')
    parser.add_argument('-fuid', '--from-uid', help='from user id')
    parser.add_argument('-tuid', '--to-uid', help='to user id')
    parser.add_argument('-p', '--post-content', help='content of fuid')
    parser.add_argument('-cfg', '--config', help='key in the calaf config file')
    args = parser.parse_args()
    
    # temporary: force to set cpu
    # device = 'cpu'

    prob = pred_influence_to_uid(from_uid=args.from_uid, to_uid=args.tuid, post_content=args.post_content, config=args.config)

    print('influence prob from : {0} to {1}: {2}'.format(args.from_uid, args.to_uid, prob))
    
