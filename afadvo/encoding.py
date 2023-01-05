import argparse
# import torch
from nn.models.vgae import VGAEEmb
from nn.models.gae import GAEEmb
from nn.models.node2vec import Node2VecEmb
from utils.database import AFPreprocessingMongoDB
import pickle
import os
import warnings

from config.training import train_info, encoding_info

# Ignore warnings
warnings.filterwarnings("ignore")

def encoding(data_id, model_info_id, model_type, device):

    # CONNECT TO AF PREPROCESSING DB
    afdb = AFPreprocessingMongoDB()

    # LOAD DATA FROM data_id
    data = pickle.loads(afdb.get_graph_data(data_id)['data'])
    
    # LOAD MODEL FROM model_info_id
    model_dt = pickle.loads(afdb.get_model_info(model_info_id)['model_info'])

    dim = model_dt['dim']
    model_weights = model_dt['weights']

    # GENERATE CORRESPONDING DATA
    print('Type of model: ', model_type)
    if model_type == 'vgae':
        model = VGAEEmb(data, dim, save_path=None, split_test=False)
        
    elif model_type == 'gae':
        model = GAEEmb(data, dim, save_path=None)
        
    elif model_type == 'node2vec' or model_type == 'n2v':
        print('Node2vec model havent been implemented yet, stop the training.')
        return 
    else:
        print('No matched mode type found')
        return None

    model.model.load_state_dict(model_weights)

    save_path = os.path.join(encoding_info['LOCAL_PATH'], data_id, model_info_id, '_embedding.pt')
    embedding = model.predict(data, device, save_emb_path=save_path)


    # SAVE EMEBDDING TO DATABASE
    oid = afdb.insert_embedding(save_path, model_info_id, data_id)
    print('- Inserted embedding to database with id: ', oid.inserted_id)

    return embedding # TO-DO: return _id

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Encoding to embedding by dedicated model')

    parser.add_argument('-gdid', '--graph_data_id', help='graph data id in AF preproccesing DB')
    parser.add_argument('-mid', '--model_info_id', help='model id from AF preprocessing DB')
    parser.add_argument('-gpu', '--gpu', default=False, help='Acclerate by GPU')

    args = parser.parse_args()

    # SET DEVICE
    device = 'gpu' if args.gpu == True else 'cpu'
    encoding(data_id=args.graph_data_id, model_info_id=args.model_info_id, model_type= encoding_info['MODEL_TYPE'], device=device)


