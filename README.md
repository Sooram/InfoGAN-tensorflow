# InfoGAN-tensorflow
Slightly changed codes from [this repository](https://github.com/AndyHsiao26/InfoGAN-Tensorflow).

## Original paper
[InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/pdf/1606.03657.pdf)

"This paper describes InfoGAN, an information-theoretic extension to the Generative Adversarial Network that is able to learn disentangled representations in a completely unsupervised manner. InfoGAN is a generative adversarial network that also maximizes the mutual information between a small subset of the latent variables and the observation."

![Overview](https://github.com/Sooram/InfoGAN-tensorflow/blob/master/network.PNG)

Basically, you add some latent codes, a set of random int(categorical) or random float(continuous), to the noise vector and the network tries to maximize the mutual information between each code and the generated image. By doing so, each of the latent variables will be mapped to some salient attributes of the input data, such as tilting or thickness of characters(MNIST). As a result, you can control some features of the generated image by manipulating the codes. 

Loss function: min(G,Q) max(D) VInfoGAN(D, G, Q) = V (D, G) − λI(c; G(z, c)) \
Mutual information: I(c; G(z, c)) = H(c) − H(c|G(z, c)) = ... ≥ Ec∼P (c),x∼G(z,c)[log Q(c|x)] + H(c) 

Maximize mutual information \
<=> Leave out all the constant factors & Maximize Q(c|x): estimate the likelihood of seeing that code for the given generated input x (Maximum Likelihood Estimation) \
<=> Minimize NLL(negative log likelihood)

Q loss = cross entropy(categorical code, Qcat(G(z, c))) + 0.1(lamda) * gaussian NLL(continuous code, Qcont(G(z, c))) \
Discriminator loss = D loss + Q loss \
Generator loss = G loss + Q loss

## Results
![Overview](https://github.com/Sooram/InfoGAN-tensorflow/blob/master/test/continuous_1_col_cat_row_change.png) thickness \
![Overview](https://github.com/Sooram/InfoGAN-tensorflow/blob/master/test/continuous_2_col_cat_row_change.png) tilt \
![Overview](https://github.com/Sooram/InfoGAN-tensorflow/blob/master/test/continuous_0_col_cat_row_change.png) \
Seems like this continuous code doesn't have any special attribute mapped. 
