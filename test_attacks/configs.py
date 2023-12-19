from attacks.fast_gradient_sign_method import fast_gradient_sign_method
from attacks.noise import noise
from attacks.momentum_iterative_method import momentum_iterative_method
from attacks.projected_gradient_descent import projected_gradient_descent
from attacks.carlini_wagner_l2 import carlini_wagner_l2
from attacks.spsa import spsa


def config_fgsm():
    return {
        'epsilon': 0.3,
        'clip_min': 0.,
        'clip_max': 1.
    }
    
    
def config_noise():
    return {
        'eps': 0.3,
        'clip_min': 0.,
        'clip_max': 1.
    }
    
    
def config_momentum_iterative_method():
    return {
        'eps': 0.3,
        'alpha': 0.1,
        'num_iter': 40,
        'decay_factor': 1.0,
        'clip_min': 0.,
        'clip_max': 1.
    }
    
def config_projected_gradient_descent():
    return {
        'eps': 0.3,
        'alpha': 0.1,
        'num_iter': 40,
        'clip_min': 0.,
        'clip_max': 1.
    }
    
def config_carlini_wagner_l2():
    return {
        'num_iter': 40,
        'lr': 0.01,
        'c': 1.0,
        'kappa': 0.0,
        'clip_min': 0.,
        'clip_max': 1.
    }
    
def config_spsa():
    return {
        'num_iter': 40,
        'lr': 0.01,
        'c': 1.0,
        'clip_min': 0.,
        'clip_max': 1.
    }

def load_attack(attack_method):
    if attack_method == 'fgsm':
        return fast_gradient_sign_method, config_fgsm()
    elif attack_method == 'noise':
        return noise, config_noise()
    elif attack_method == 'momentum_iterative_method':
        return momentum_iterative_method, config_momentum_iterative_method()
    elif attack_method == 'projected_gradient_descent':
        return projected_gradient_descent, config_projected_gradient_descent()
    elif attack_method == 'carlini_wagner_l2':
        return carlini_wagner_l2, config_carlini_wagner_l2()
    elif attack_method == 'spsa':
        return spsa, config_spsa()
    else:
        raise ValueError('attack method not recognised')
    