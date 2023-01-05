import numpy as np
import sklearn
from sklearn import metrics
import torch
import pickle



def get_similarity_2(uid, user_edge_list, z, node_dict, threshold=0.5, limit=100):
    stack = []
    
    af_directions = {}
    af_probs = {}           
    af_degs = {}

    if type(z) is str:
        z = torch.load(z)
    
    stack.append(str(uid))
    counter = 0
    while len(stack) > 0 and counter < limit:
        counter += 1
#         print('stack: ', stack)
        node = stack.pop()
        
        if str(node) in stack:
            print('***********', node, '************')
            
        # look-up in user_edge_list
        try:
            uids = user_edge_list[str(node)]
        except KeyError:
            print('- Key ', node, ' finished')
            continue
        
        # cal similarity
        if len(uids) == 0:
            continue

        if len(uids) == 1 and node == uids[0]:
            continue

        # if node not in uids
        index = np.where(uids == str(node))[0]
        if len(index) == 0:
            t = np.concatenate([np.array([node]), uids])
        else:
            index = index[0]
            t = np.concatenate([np.array([uids[index]]), uids[:index], uids[index+1:]])

        atts = np.array([np.array(z[node_dict[int(k)], :]) for k in t])
        sims = metrics.pairwise.cosine_similarity(atts)[0,:]
        
        edge_list = []
        edge_prob = []
        edge_outdeg = []
        for i in range(1, len(t)):
            if sims[i] > threshold:
                
                edge_list.append(t[i])
                edge_prob.append(sims[i])
                if not str(t[i]) in user_edge_list:
                    edge_outdeg.append(1)
                    print('- End at ', str(t[i]))
                else:
                    edge_outdeg.append(len(user_edge_list[str(t[i])]))
                    # ADD TO STACK
                    stack.append(str(t[i]))
                    
                
        af_directions[node] = edge_list
        af_probs[node] = edge_prob
        af_degs[node] = edge_outdeg
        
        # print(af_directions)
    
    return af_directions, af_probs, af_degs


def cal_af_2(uid, af_directions, af_probs, af_degs):
    af_score = 0.0
    
    for i, (k,v) in enumerate(af_directions.items()):
        for j in range(len(v)):
            af_score += (af_probs[k][j] * af_degs[k][j])
    return af_score

def predict_embedding(model, data, normalized=True, gpu=True, save_path=None):
    r'''
    Encode to embedding

    Args:
        model: str or pytorch model
        data: `torch_geometric.data.Data`
        gpu: boolean
        save_path: str

    :rtype: 
        `torch.Tensor` embedding
    '''

    if type(model) is str:  
        model = torch.load(model)

    if type(data) is str:
        data = torch.load(data)

    if normalized:
        scaler = sklearn.preprocessing.StandardScaler()
        data.x = torch.from_numpy(scaler.fit_transform(data.x))
        data.edge_attr = (0.6*data.edge_attr[:,0] + 0.3*data.edge_attr[:,1] + 0.1*data.edge_attr[:,2])

    if gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model = model.to(device)
    z = model.encode(data.x.float().to(device), data.edge_index.to(device), data.edge_attr.to(device))

    if not save_path is None:
        pickle.dump(z, open(save_path, 'wb'))

    return z



def predict_post_influence(from_uid_embedding, to_uid_embedding, post_embedding, csif_model, device=torch.device('cpu')):
    r'''
    Predict the diffusion of a given post from 1 user to 1 other user.
    Return the probability (between 0 and 1)
    
    Args:
        from_uid_embedding: torch tensor, user id who post the post
        to_uid_embedding: torch tensor, user id to be predicted
        post_embedding: torch tensor, embedding
        csif_model: Pytorch model, content-social influence model
    '''

    u_pair = torch.cat([from_uid_embedding, to_uid_embedding])

    csif_model.eval()
    prob, _, _ = csif_model.predict(u_pair, post_embedding, device=device)
    prob = prob.detach().sigmoid().numpy()[0][0]

    return prob