# README 

## To run the attacks


## Summary of GBZ

We consider the GBZ settings, we provide below a more detailed explanation on the implementation and the caclulation of a the loss function. Other settings can be implemeented in a similar fashion. The GBZ setting assumes the existance of a latent variable $z$ such that the image $x$ and the label $y$ can be respectively modelled by $p(x|z)$ and $p(y|z)$. 

We now detail provide the implementation details for this notebook. For neural networks we denote the hidden dimension by dimH. Similarly dimZ is the latent variable dimension and dimY is the dimension of one-hot encoded labels i.e. dimY is equal to the number of classes. We assume that an image $x$ is a square pixel gride of length dimX. 

The implementation of GBZ relies on four neural networks with two denoted as decoders and two as encoders. We list the diffent neural networks below:
* An encoder that aims to approximate $p(z|x,y)$. This encoder is made up of two complenents:
    * A CNN $e^1_\theta$ which takes an image $x$ and output a vectore in dimension dimH
    * An MLP $e^2_\theta$ which takes the output of $e^1_\theta$ and a label $y$ to model the latent varibale $z$. More precisely, we use a probabilistic setting such that $\mu, \log(\Sigma) = e^2_\theta(e^1_\theta(x),y)$ and $z\sim \mathcal{N}(\mu, \Sigma)$. Here the $\log$ is used for numerical stability
* A CNN decoder $d^1_\theta(z)$ mapping an latent variable $z$ to an image $x$. This network is implemented as a flipped CNN i.e. a neural network with the head made up of an MLP and followed by a CNN in the final layers. This network aims to approximate $p(y|z)$.
* An MLP decoder $d^2_\theta(z)$ that maps from dimZ to dimY i.e converts a latent variable to a distribution over the labels. This aim to approximate $p(y|z)$

We now descirbe how the training is done over a single example $(x,y)$. This will be applied in the notebook in a minibatch strategy for better computational performance on the parameters.
1. Get the encoding $e= e^1_\theta(x)$
1. Calculate the posterior $\mu, \log(\Sigma) = e^2_\theta(e, y)$
1. sample $K$ random  *posterior* latent varibales $z^1, \ldots,z^K\sim \mathcal{N}(\mu , \Sigma)$
1. calculate the log-likelihood (under the posterior) $q^j = \log(f_{\mathcal{N}(\mu, \Sigma)}(z^j))$ 
1. calculate the log-likelihood (under the gaussian prior) $p^j = \log(f_{\mathcal{N}(0,I)}(z^j))$ 
1. Use the decoder to reconstruct the images $\hat{x}^i=d^1(z^i)$
1. Use the decoder to reconstruct the labels $\hat{y}^i=d^2(z^i)$
1. Calcuale the MSE loss of reconstructured images $l_1^j= ({\hat{x}^j}-x)^2$
1. Claculate the cross-entropy loss of reconstructed labels $l_2^j = \text{CE}(y,\hat{y}^j) $
1. Calculate the total loss $$b^j= (\beta-1) l_1^j+l_2^j+p^j$$
1. Average over the $K$ samples, using a logsumexp function

## Summary of DBX (MODEL E)

## Summary of DFZ


