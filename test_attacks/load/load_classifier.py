import torch
import numpy as np

def load_classifier(model_name, data_name):
    if 'bayes' in model_name or model_name == 'GBZ':
        from test_attacks.load.load_bayes_classifier import BayesModel
        if data_name == 'mnist':
            dimZ = 64
        else:
            dimZ = 128
        hidden_channels = 64
        model = BayesModel(data_name, hidden_channels, dimZ=dimZ)    
    else:
        raise ValueError('classifier type not recognised')     

    return model

