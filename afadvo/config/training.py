

train_info = {
    'MODEL_TYPE': 'vgae',
    'USE_GPU': True,
    'EPOCHS': 5000,
    'EMBEDDING_DIMENSION': 64,
    'OPTIMIZER': 'adam',
    'LEARNING_RATE': 0.01,
    'MONITOR': 'ap', # average precision
    'TEMP_CHECKPOINT': './temp/temp_model_weights.pt'
}

encoding_info = {
    'LOCAL_PATH': 'files/embeddings/',
    'MODEL_TYPE': 'vgae'
}