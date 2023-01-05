import pymongo
from pymongo import MongoClient
import numpy as np
import pandas as pd


def get_kapi_db(ip='54.169.171.51:27017', username='kapiReadOnly', password='pl2oieAt9#tnWV!Yc0', authSource='kapi', authMechanism='SCRAM-SHA-1'):
    client = MongoClient(ip, username=username, password=password, authSource=authSource, authMechanism=authMechanism)
    
    fdb = client['kapi']
    
    return fdb

def get_af_preprocessing_db(ip='54.169.171.51:27017', username='dsdReadWrite', password='ZVzVZ9VgR-eKnqh', authSource='ds-data', authMechanism='SCRAM-SHA-1'):
    client = MongoClient(ip, username=username, password=password, authSource=authSource, authMechanism=authMechanism)
    
    fdb = client['ds-data']

    return fdb

def get_af_files_collection(ip='54.169.171.51:27017', username='dsdReadWrite', password='ZVzVZ9VgR-eKnqh', authSource='ds-data', authMechanism='SCRAM-SHA-1'):
    client = MongoClient(ip, username=username, password=password, authSource=authSource, authMechanism=authMechanism)
    
    fdb = client['ds-data']

    return fdb['AF_files']