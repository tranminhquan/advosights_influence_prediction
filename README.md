# Influence Prediction

# Overview

Amplification factor score version 1 is based on **Knowledge Graph** and **Deep Learning**. As a result, it requires data and training to accomplish.

In particular, there are following steps to calculate the AF:
1. **Transforming data**: transform raw data (from database) to structural knowledge graph data given a list of users
2. **Training model**: create the model *based on our approach*, then, the transformed data is fed to the model
3. **Encoding embedding**: Given list of users, the model encodes the corresponding data to embedding vectors. These vectos are applied to calculate the AF score
4. **Calculating AF score**: Calculate the AF score based on the embedding vector of users and our algorithm
5. **Predict post's influence**: Given a content, predict which users will be influenced

# How to run

## Transfrom data

Run 
`python transform-graph.py --uidlist= --from_date= --to_date`

* `uidlist`: list of user ids on the database
* `from_date`: date to start getting the data
* `to_date`: date to complete getting the data

The `transform-graph` function produces $4$ id from the database corresponding to *edge list, node attribute, edge attribute* and *torch graph data*, respectively.

## Train model  
To train the model, run

`python train.py --graph_data_id= `

* `graph_data_id`: the id of *torch graph data*

After training, the `train` function produces the weights of model stored in database and return the id

## Encode
To encode the dedicated list of user ids, run

`python encoding.py --graph_data_id=  --model_info_id=  --gpu=  `

* `graph_data_id`: id of *torch graph data*
* `model_info_id`: if of trained model's weights
* `gpu`: `True/False` - whether to acclerate the encoding by GPU or not

After encoding, the function produces the embedding id

## Calculate AF
`python calaf.py --uid=  --embedding_id=  --gpu= `

* `uid`: user id to calculate the AF score 
* `embedding_id`: embedding id that contains the `uid`
* `gpu`: `True/False` - whether to acclerate the calculation by GPU or not

# Run showcase demo

- Env: ubuntu 20.04
- prepare to run at local development
- Update system: `sudo apt-get update`
- Install packages: `sudo apt-get install gfortran libopenblas-dev liblapack-dev python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools python3-virtualenv`
- Make new environment: `python -m virtualenv v`
- Active the enviroment: `source v/bin/active`
- Install packages: `pip install -r requirements.txt`

- cd to `vizdemo` folder: `cd afadvo/vizdemo`
- run `app.py` file `python app.py`



