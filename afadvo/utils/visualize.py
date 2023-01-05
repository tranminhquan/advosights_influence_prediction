import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def visualize_tsne(z, n_dim=2, label_dict=None, figsize=(150,150), s=20, save_emb_path=None, save_fig_path=None):
    '''
        Visualize vector space by given alogrithm, if label_dict is given, then plot scatter by their colors
        '''
    embs = TSNE(n_components=n_dim).fit_transform(z.cpu().detach().numpy())

    with open(save_emb_path, 'wb') as dt:
        pickle.dump(embs, dt)

    # visualize
    plt.figure(figsize=figsize)
    
    if label_dict is None:
        plt.scatter(embs[:, 0], embs[:, 1], s, label='user')
    else:
        except_ids = np.empty(0)
        for k,v in label_dict.items():
            except_ids = np.concatenate([except_ids, v])
            plt.scatter(embs[v,0], embs[v,1], s + 200, label=str(k))
            
        rest_ids = np.delete(np.arange(len(z)), except_ids)
        plt.scatter(embs[rest_ids, 0], embs[rest_ids, 1], s, label='the rest')

    plt.savefig(save_fig_path)
    plt.legend()
    plt.show()
    
    return embs

def plot_af_user(uid, af_directions, tsne_emb, node_dict, s=20, figsize=(50,50)):
    '''Plot the influencer and their "fans" '''
    if type(tsne_emb) is str:
        with open(tsne_emb, 'rb') as dt:
            tsne_emb = pickle.load(dt)
    
    keys = np.unique(list(af_directions.keys()))
    values = np.unique(np.concatenate(list(af_directions.values())))
    uids = np.concatenate([keys, values])
    
    uids = np.array([node_dict[int(k)] for k in uids])
    
    main_uid = node_dict[int(uid)]
    
    plt.figure(figsize=figsize)
    plt.scatter(tsne_emb[:,0], tsne_emb[:,1], color='blue')
    plt.scatter(tsne_emb[uids,0], tsne_emb[uids,1], color='orange', s=1500)
    plt.scatter(tsne_emb[main_uid,0], tsne_emb[main_uid,1], color='red', s=3000)
    
    plt.show()
    
