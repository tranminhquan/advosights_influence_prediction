# AF score script | User embedding with node embedding
---
R: QuanTran

Modification date:
* 4 - Jun - 2020 (v2.0)
* 5- Jun - 2020 (v2.1)

*Embedding from user (node) in knowledge graph to laten space (vector space). By embedding to vector space, user's attributes can be extracted to become feature vectors, which are significantly helpful for deep learning. This algorithm is implemented by [Torch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html), a Pytorch-based specific algorithm for Knowledge Graph data significantly enhancing on GPU.*

---
## Prequistite
* `user_edge_list`: python dictionary in edge_list data structure
* `node_atts`: users' attributes
* `edge_atts`: attributes of user's interaction
---
## Agenda

1. [Generate Data toch-geometric](#generate_data)
2. [Node embedding methods](#node_embedding):  
    * [Node2Vec](#node2vec)
    * [Graph auto-encoder](#gae)
    * [Variational Graph auto-encoder](#vgae)

## Prev. step:
* [Transfrom from raw data into structural knowledge graph data](http://14.161.9.65:9012/notebooks/Quan/Advosights/adv/sources/AF_score_transform_data.ipynb#)

## Nextstep:
* [Word embedding for users' posts]()