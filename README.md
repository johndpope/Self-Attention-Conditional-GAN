# Self-Attention-Conditional-GAN
Pytorch Implementation of Self-Attention Conditional GAN to generate dog images.

This implementation is initially aimed to run on Dog images from the Kaggle competition:  
https://www.kaggle.com/c/generative-dog-images/  

The dog dataset can be downleaded from the link above, the two files all-dogs, annotations should be placed in the main directory.  

Many parts of the code are not mine but assembled from different sources online and kernels including:  
1- https://www.kaggle.com/tikutiku/gan-dogs-starter-biggan  
2- https://www.kaggle.com/yukia18/sub-rals-biggan-with-auxiliary-classifier  
3- https://github.com/christiancosgrove/pytorch-spectral-normalization-gan  
4- https://github.com/pfnet-research/sngan_projection  

The implementation uses:  
1- Spectral normalization: https://arxiv.org/abs/1802.05957  
2- Conditional Batch normalization (in the generator): https://arxiv.org/pdf/1707.00683.pdf   
3- cgan with projection discriminator: https://arxiv.org/abs/1802.05637  
4- Self-Attention GAN: https://arxiv.org/pdf/1805.08318.pdf  
5- GANs Trained by a Two Time-Scale Update Rule : https://arxiv.org/abs/1706.08500  
6- Some ideas from BigGAN: https://arxiv.org/abs/1809.11096  


