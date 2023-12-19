import torch

def log_gaussian_prob(x, mu, log_sig):
    logprob = -(0.5 * torch.log(2 * torch.pi) + log_sig) \
                - 0.5 * ((x - mu) / torch.exp(log_sig)) ** 2
    return torch.sum(logprob, dim=1)

def sample_gaussian(mu, log_sig, K):
    mu = mu.repeat(K, 1)
    log_sig = log_sig.repeat(K, 1)
    z = mu + torch.exp(log_sig) * torch.randn_like(mu)
    return mu, log_sig, z

def sample_gaussian_fix_randomness(mu, log_sig, K, seed):
    torch.manual_seed(seed)
    mu = mu.repeat(K, 1)
    log_sig = log_sig.repeat(K, 1)
    z = mu + torch.exp(log_sig) * torch.randn_like(mu)
    return mu, log_sig, z

def encoding(enc_mlp, fea, y, K, use_mean, fix_samples, seed):
    mu_qz, log_sig_qz = enc_mlp(fea, y)
    if use_mean:
        z = mu_qz
    elif fix_samples:
        _, _, z = sample_gaussian_fix_randomness(mu_qz, log_sig_qz, K, seed)
    else:
        _, _, z = sample_gaussian(mu_qz, log_sig_qz, K)
    logq = log_gaussian_prob(z, mu_qz, log_sig_qz)
    return z, logq

def lowerbound(x, fea, y, enc_mlp, dec, ll, K=1, IS=False, 
               use_mean=False, fix_samples=False, seed=0, z=None, beta=1.0):
    if use_mean:
        K = 1
        fix_samples=False

    if z is None:
        z, logq = encoding(enc_mlp, fea, y, K, use_mean, fix_samples, seed)
    else:
        mu_qz, log_sig_qz = enc_mlp(fea, y)
        logq = log_gaussian_prob(z, mu_qz, log_sig_qz)

    x_rep = x.repeat(K, 1, 1, 1)
    y_rep = y.repeat(K, 1)

    # prior
    log_prior_z = log_gaussian_prob(z, 0.0, 0.0)

    # decoders
    pyz, pxz = dec
    mu_x = pxz(z)
    if ll == 'bernoulli':
        logp = torch.distributions.Bernoulli(mu_x).log_prob(x_rep).sum(dim=1)
    elif ll == 'l2':
        logp = -torch.sum((x_rep - mu_x)**2, dim=list(range(1, len(x_rep.shape))))
    elif ll == 'l1':
        logp = -torch.sum(torch.abs(x_rep - mu_x), dim=list(range(1, len(x_rep.shape))))

    logit_y = pyz(z)
    log_pyz = -torch.nn.functional.cross_entropy(logit_y, y_rep, reduction='none')

    bound = logp * beta + log_pyz + (log_prior_z - logq)
    if IS and K > 1:	# importance sampling estimate
        N = x.shape[0]
        bound = bound.view(K, N)
        bound = torch.logsumexp(bound, dim=0) - torch.log(float(K))

    return bound
