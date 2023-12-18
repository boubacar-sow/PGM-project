import torch
import numpy as np

def load_classifier(model_name, data_name, path=None, attack_snapshot=False):
    if 'bayes' in model_name:
        from load_bayes_classifier import BayesModel
        conv = True
        vae_type = model_name[-1]
        use_mean = False
        fix_samples = False
        K = int(model_name.split('_')[1][1:])
        if conv:
            model_name += '_cnn'
        else:
            model_name += '_mlp'
        if 'Z' in model_name:
            dimZ = int(model_name.split('_')[2][1:])
        else:
            if data_name == 'mnist':
                dimZ = 64
            else:
                dimZ = 128
        model = BayesModel(data_name, vae_type, conv, K, 
                           attack_snapshot=attack_snapshot, use_mean=use_mean, fix_samples=fix_samples,
                           dimZ=dimZ)   
    elif 'fea' in model_name:
        from load_bayes_classifier_on_fea import BayesModel
        conv = True
        _, K, vae_type, fea_layer = model_name.split('_')
        K = int(K[1:])
        use_mean = False
        fix_samples = False
        if conv:
            model_name += '_cnn'
        else:
            model_name += '_mlp'
        if 'Z' in model_name:
            dimZ = int(model_name.split('_')[2][1:])
        else:
            if data_name == 'mnist':
                dimZ = 64
            else:
                dimZ = 128
        model = BayesModel(data_name, vae_type, fea_layer, conv, K, 
                           attack_snapshot=attack_snapshot, use_mean=use_mean, fix_samples=fix_samples,
                           dimZ=dimZ)   
    else:
        raise ValueError('classifier type not recognised')     

    return model
