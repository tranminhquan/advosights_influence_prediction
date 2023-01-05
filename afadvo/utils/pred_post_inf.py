'''
    predict the diffusion of a given post from 1 user to 1 other user
'''

import torch
import pickle
from utils.predict import predict_post_influence
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
from config.calaf import *
from nn.models.csif import *
from nn.models.hiersoftmax import *
from collections import OrderedDict

class PostInfluencePredict():

    def load_files(self, dataset):
        self.dataset = dataset.lower()

        self.csif_model = POST_PRED[self.dataset]['CSIF_MODEL']
        self.user_edge_list = POST_PRED[self.dataset]['USER_EDGE_LIST']
        self.node_dict = POST_PRED[self.dataset]['NODE_DICT']
        self.user_embeddings = POST_PRED[self.dataset]['USER_EMBEDDINGS']
        self.w2v_model = POST_PRED[self.dataset]['W2V_MODEL']

        self.csif_model = torch.load(self.csif_model, map_location='cpu') if type(self.csif_model) is str else self.csif_model
        self.user_edge_list = pickle.load(open(self.user_edge_list, 'rb')) if type(self.user_edge_list) is str else self.user_edge_list
        
        self.node_dict = pickle.load(open(self.node_dict, 'rb')) if type(self.node_dict) is str else self.node_dict
        self.node_dict = {str(k):v for k,v in self.node_dict.items()}

        self.user_embeddings = pickle.load(open(self.user_embeddings, 'rb')) if type(self.user_embeddings) is str else self.user_embeddings
        self.user_embeddings = torch.from_numpy(self.user_embeddings)

        self.w2v_model = Doc2Vec.load(self.w2v_model) if type(self.w2v_model) is str else self.w2v_model

    def predict(self, from_uid, to_uid, post_content):
        r'''
            Predict the probability of influence, given 2 users and a post content
        '''

        # TODO: suong.hoang
        post_embedding = word_tokenize(post_content.lower())
        post_embedding = self.w2v_model.infer_vector(post_embedding)


        # get embedding from uid
        from_uid_embedding = self.user_embeddings[self.node_dict[from_uid]]
        to_uid_embedding = self.user_embeddings[self.node_dict[to_uid]]


        # predict
        print('predict post')
        prob = predict_post_influence(from_uid_embedding, to_uid_embedding, post_embedding, self.csif_model)

        print('* Predict post influence from ', from_uid, ' to ', to_uid, ' : ', prob)

        return prob

    def generate_transition_matrix(self, users, device=torch.device('cpu')):
        '''
        Generate transition matrix R based on users and the given post embedding
        ---
        Args:
            users: uid list of all users in community
            csif_model: CSIF model
            user_edge_list: 
            node_dict: dictionary mapping users to indices in user embeddings
            user_embeddings: embeddings from phase 1
            post_emebdding: embedding of post to calculate the probagation
            
        Return:
            Transition matrix R
        '''
        fails = 0
        
        n_users = len(users)
        trans_matrix = torch.zeros((n_users, n_users), dtype=torch.float)
        
        for user in users:
            if user in self.user_edge_list:
                # get all influenced users
                inf_users = self.user_edge_list[user]
                
                try:
                    # get indices
                    uindex = self.node_dict[str(user)]
                    inf_uindices = [self.node_dict[str(k)] for k in inf_users]

                    # loop in trans_matrix
                    for i in inf_uindices:
                        upair = np.array([self.user_embeddings[uindex], self.user_embeddings[i]]).flatten()
                        temp, _, _ = self.csif_model.predict(upair, self.post_embedding)
                        temp = temp.detach().cpu().sigmoid()[0][0]
                        if temp > 0.0:
                            trans_matrix[uindex][i] = temp
                except:
                    fails += 1
                    continue

        # print('* Unidentify users rate: ', fails / len(users))
                    
        return trans_matrix, fails

    def pagerank(self, R, alpha=0.85, epochs=100):
        r'''
            Cal pagerank given trans. matrix R
        '''
        # re-exportation matrix v
        n_nodes = R.size(0)

        rex_vec = torch.ones(n_nodes, dtype=torch.float)
        rex_vec = rex_vec / n_nodes
        rex_vec = torch.unsqueeze(rex_vec, dim=-1)

        s = torch.ones(n_nodes, dtype=torch.float)
        s = s / n_nodes

        s = torch.unsqueeze(s, dim=-1)

        #     return transition_matrix, s
        rs = s

        # cal pagerank
        # print('Cal PageRank ...')

        min_dist = 10000.

        n_epochs = 0
        for i in range(epochs):
            n_epochs += 1

            s = (alpha*(torch.mm(R,s) / n_nodes)) + ((1-alpha)*rex_vec)

            # print('distance: ', torch.mean(rs - s))
            # print('s sum: ', s.sum())

            if (rs == s).all():
                print('Converage.')
                return s

            rs = s
            # print('***********************')
            
        return rs

    def get_top_by_post(self, top=10, uids=None, alpha=0.85, device=torch.device('cpu')):
        '''
        get top-K influencer by a given post
        '''
        if uids is None: # choose all uid in model
            uids = list(self.node_dict.keys())

        uids = [str(k) for k in uids]

        # cal transition matrix
        R, fails = self.generate_transition_matrix(uids, device=device)
        
        # cal pagerank
        ranks = self.pagerank(R, alpha)
        ranks = ranks[:,0]
        
        # create dict
        rank_dict = OrderedDict(dict(zip(uids, ranks)))
        rank_dict = {k: v for k, v in sorted(rank_dict.items(), key=lambda item: item[1], reverse=True)}

        rank_uids = list(rank_dict.keys())[:top]
        
        return rank_uids


