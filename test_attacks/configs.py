from project.attacks import fast_gradient_method, projected_gradient_descent, momentum_iterative_method, carlini_wagner_l2, spsa, noise 

def config_fgsm():
    return {
        'eps': 0.3,
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
