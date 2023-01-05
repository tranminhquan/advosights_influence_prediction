'''Training embedding model (phase 1)'''
import argparse
# import torch
from nn.models.vgae import VGAEEmb
from nn.models.gae import GAEEmb
from nn.models.node2vec import Node2VecEmb
from utils.database import AFPreprocessingMongoDB
import pickle

import warnings

from config.training import train_info

# Ignore warnings
warnings.filterwarnings("ignore")

def train(model_type, data_id, dim, gpu, epochs, optimizer, learningrate, monitor, checkpoint):

    # SET DEVICE
    device = 'gpu' if gpu == True else 'cpu'

    # LOAD GRAPH DATA FROM AF PREPROCESSING DATA
    print('- Load data from AF Preprocessing DB')
    afdb = AFPreprocessingMongoDB()
    graph_data = afdb.get_graph_data(data_id)
    data = pickle.loads(graph_data['data'])

    print(data)
    if data is None:
        print('* GRAPH DATA NOT FOUND, stop training')
        return None

    # TRAIN
    print('Type of model: ', model_type)
    if model_type == 'vgae':
        model = VGAEEmb(data, dim, save_path=checkpoint, split_test=False)
        history, model_info = model.train(int(epochs), device, optimizer, lr=learningrate, monitor=monitor, return_optimal=True)
    elif model_type == 'gae':
        model = GAEEmb(data, dim, save_path=checkpoint)
        history = model.train(int(epochs), device, optimizer, lr=learningrate, monitor=monitor)
    elif model_type == 'node2vec' or model_type == 'n2v':
        print('Node2vec model havent been implemented yet, stop the training.')
        return 
    else:
        print('No matched mode type found')
        return None
        
    
    # INSERTED TO MONGO DB
    model_oid = afdb.insert_model_info(model_info, data_id, history)

    print('Inserted model DONE with id: ', model_oid.inserted_id)
    
    return model_oid.inserted_id

# def train(model_type, data, savepath, dim, gpu, epochs, optimizer, learningrate, monitor, historypath):

#     # setting device
#     # if gpu is True:
#     #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     #     if str(device) != 'cuda':
#     #         print('GPU is not available, switch to CPU')
#     #     else:
#     #         print('GPU is activated')
#     # else:
#     #     device = torch.device('cpu')

#     device = 'gpu' if gpu == True else 'cpu'

#     print('Type of model: ', model_type)
#     history = None
#     if model_type == 'vgae':
#         model = VGAEEmb(data, dim, savepath)
#         history = model.train(int(epochs), device, optimizer, lr=learningrate, monitor=monitor)
#     elif model_type == 'gae':
#         model = GAEEmb(data, dim, savepath)
#         history = model.train(int(epochs), device, optimizer, lr=learningrate, monitor=monitor)
#     elif model_type == 'node2vec' or model_type == 'n2v':
#         print('Node2vec model havent been implemented yet, stop the training.')
#         return 
#     else:
#         print('No matched mode type found')

#     if not historypath is None and not history is None:
#         print('-- Saving training history')
#         with open(historypath, 'wb') as dt:
#             pickle.dump(history, dt)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training node embedding model for users')

    parser.add_argument('-gdid', '--graph_data_id', default=None, help='graph data id in AF preproccesing DB')

    args = parser.parse_args()

    train(train_info['MODEL_TYPE'], args.graph_data_id, 
            train_info['EMBEDDING_DIMENSION'], train_info['USE_GPU'], train_info['EPOCHS'], 
            train_info['OPTIMIZER'], train_info['LEARNING_RATE'], train_info['MONITOR'],
            train_info['TEMP_CHECKPOINT'])

    
