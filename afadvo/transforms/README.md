# AF score script | Transform from raw data into structural knowledge graph data
---
R: QuanTran

Modification date: 
* 25 - May - 2020 (v.2.0)
* 1 - Jun - 2020 (v.2.1)

*Transfrom from raw data into structural knowledge graph data*



---

## Agenda
1. [Prerequisite](#prerequisite)
2. [Knowledge graph data structure](#knowledge_graph_data_structure)  
    2.1. [Edge list](#edge_list)  
    2.2. [User attribute (Node attribute)](#node_attribute)  
    2.3. [Edge attribute](#edge_attribute)  
3. [Transformation process](#transformation)  
    3.1. [Building edge list](#building_edge_list)  
    3.2. [Building node attribute](#building_node_attribute)  
    3.3. [Building edge attribute](#building_edge_attribute)  
    
4. [Code](#code)  


## Nextstep:
Two options for Node Embedding
* [Node2vec embedding with Pytorch](http://14.161.9.65:9012/tree/Quan/Advosights/adv/sources/AF_scire_GG.ipynb) (*Temporary official*)
* [Graph auto-encoder with Pytorch](http://14.161.9.65:9012/notebooks/Quan/Advosights/adv/sources/AF_score%20graph%20autoencoder.ipynb) (*On-going research*)
---

---

## 1. Prerequisite <a id='prerequisite'></a>

* `chosen_users.csv`: Pandas Dataframe of concerned users for graph
* `chosen_post_2.csv`: Pandas Dataframe of concerend posts
* `chosen_comment.csv`: Pandas Dataframe of concerned comments related to our knowledge graph
* `chosen_react.csv`: Pandas Dataframe of concerned reactions related to our knowledge graph

*All files came from [the preprocessing script](http://14.161.9.65:9012/notebooks/Quan/Advosights/adv/sources/AF_score_data_analytics.ipynb)*

---

## 2. Knowledge graph data structure <a id='knowledge_graph_data_structure'> </a>

Building graph data contains the following data structures: **edge list, node attribute, edge attribute**. All are structured by Python dictionary type.

### Edge list <a id='edge_list'></a>

Python dictionary with type 
```
{
    user_id_A: [user_id_1, user_id_7, ...],
    user_id_B: [user_id_3, user_id_4, ...],
    user_id_C: [user_id_1, user_id_5, ...],
    ...
}

```


Keys of edge list are all distincted user ids, and the corresponding values are those users who are influenced.

*For example, `user_id_A` influences on those users `user_id_1`, `user_id_2`, ....In particular, we officially defined the influence based on whether `user_id_1`, `user_id_2`, ... react, comment, share on posts of `user_id_A`*


### User attribute (Node attribute) <a id='node_attribute'></a>

Every identical users (nodes) in our graph have their own attributes (informations).  

> For the current version, we mostly concern on: **total_follower, total_friend, books_count, films_count, music_count, restaurants_count, sex** 

Data structure:
```
    {
        user_id_A: [ attrA_1, attrA_2, ..., attrA_n],
        user_id_B: [ attrB_1, attrB_2, ..., attrB_n],
        user_id_C: [ attrC_1, attrC_2, ..., attrC_n],
        ...
    }
    
```

Keys are identical user ids, values are numpy array of corresponding attributes in numerical data type

### Edge attribute <a id='edge_attribute'></a>

We arm to build weighted knowledge graph which is each edge are weighted by their own attributes.  

> For the current version, we offically define the edge attribute belongs to the number of reactions, comments and shares
  
  

*For example: user B reacted, commented or shared some posts from A, we summary that B do $10$ shares, $20$ comments and $125$ reactions from some posts of A, then*

$$
A \longrightarrow [20, 20, 125] \longrightarrow B
$$

We finally ends up with the following structure:
```
    {
        user_id_A: [ [n_shares, n_cmts, n_reacts], [n_shares, n_cmts, n_reacts], ...],
        user_id_B: [ [n_shares, n_cmts, n_reacts], [n_shares, n_cmts, n_reacts], ...],
        user_id_C: [ [n_shares, n_cmts, n_reacts], [n_shares, n_cmts, n_reacts], ...],
        ...
    }
    
```

The orders of keys and corresponding values absolutely respect to ones of **edge list** data strucutre.

*For example, values `user_id_A` is a Python list contains multiple numpy arrays describing number of shares, comments, reactions respectively of corresponding indices from edge list. That is, first numpy array belongs to `user_id_1`, the second one belones to `user_id_7`, ...*

---

## 3. Transformation process <a id='transformation'></a>

All prerequisited files are csv, the idea to transfer is straight-forward as bellows:

### Building edge list <a id='building_edge_list'></a>

#### 1. Build dictionary of `user_id - post_ids`

which means we are going to group our post ids by user ids. In other words, we want to know **who users commit which posts?**

Data structure:
```
    {
        user_id_A: [post_idA_1, post_idA_2, ...], 
        user_id_B: [post_idB_1, post_idB_2, ...], 
        user_id_C: [post_idC_1, post_idC_2, ...], 
        ...
    }
```

#### 2. Build dictionary of `post_id - share user ids`

which means we group our user ids who share by post ids. **Which posts are shared by who users?**

Data structure:
```
    {
        post_id_A: [user_id_1, user_id_2, ...], 
        post_id_B: [user_id_3, user_id_7, ...], 
        post_id_C: [user_id_3, user_id_14, ...], 
        ...
    }
```

#### 3. Building dictionary of `post_id - comment user ids`

we means we group our user ids who comment by post_ids. **Which posts are commented by who users?**

Data structure:
```
    {
        post_id_A: [user_id_1, user_id_2, ...], 
        post_id_B: [user_id_3, user_id_7, ...], 
        post_id_C: [user_id_3, user_id_14, ...], 
        ...
    }
```

#### 4. Building dictionary of `post_id - reaction user ids`

we means we group our user ids who react by post_ids. **Which posts are reacted by who users?**

Data structure:
```
    {
        post_id_A: [user_id_1, user_id_2, ...], 
        post_id_B: [user_id_3, user_id_7, ...], 
        post_id_C: [user_id_3, user_id_14, ...], 
        ...
    }
```

#### 5. Matching to build edge list

For each post id of identical user id, find those users who share, comment, react on that post, then we have the edge list of that user id

#### 6. Example

`user_id - post_ids`

```
    {
        '02542465': ['01255_21254', '12354_13244'],
        '12357846': ['12155_12545', '12485_54877'],
    }
```

`post_id - user_ids share`

```
    {
        '01255_21254': ['12357846', '457851'],
        '12354_13244': ['124556'],
        '12155_12545': ['254635'],
        '12485_54877': []
    }
```

`post_id - user_ids comment`

```
    {
        '01255_21254': ['457851'],
        '12354_13244': [],
        '12155_12545': ['124556', '254635'],
        '12485_54877': []
    }
```

`post_id - user_ids react`

```
    {
        '01255_21254': [],
        '12354_13244': ['254635'],
        '12155_12545': [],
        '12485_54877': []
    }
```

As a result, the **edge list** is:

```
    '02542465': ['12357846', '457851'],
    '12357846': ['254635', '124556', '254635']
```

### Building node attribute <a id='building_node_attribute'></a>

Get all unique user ids from **edge list**, then query from `chosen_users.csv` to get final attributes in numeric. As being concerned, we focus on **total_follower, total_friend, books_count, films_count, music_count, restaurants_count, sex**.

### Building edge attriubte <a id='building_edge_attribute'></a>

We build $3$ data structures: `edge_share`, `edge_comment`, `edge_react`

> The main idea to execute:
    1. Loop through key- values `k,v` in `edge_list`
    2. Get all post ids of `k` in `user id - post ids`
    3. Query in data frame: depends on particular dataframe (share, comment, react), represented by Python dictionary
    
Data structure:
```
{
    user_id_A: [n_share/cmt/react of user_idA_1, n_share/cmt/react of user_idA_2, ...],
    user_id_B: [n_share/cmt/react of user_idB_1, n_share/cmt/react of user_idB_2, ...],
    user_id_C: [n_share/cmt/react of user_idC_1, n_share/cmt/react of user_idC_2, ...],
    ...
}
```

#### 1. `edge_share`
Query from `chosen_post_2.csv`: *from user is in `v` and parent id is in `post ids`, group by from user on fid, count*

#### 2. `edge_comment`
Query from `chosen_comment.csv`: *from user is in `v` and post id is in `post ids`, group by from user on post id, count*

#### 3. `edge_react`
Query from `chosen_react.csv`: *from from user id is in `v` and fid is in `post ids`, group by from user id on fid, count*


#### Concatenate
From above dictionaries, concatente share, comment and reaction based on their indices