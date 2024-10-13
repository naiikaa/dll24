# DDPM
## Introduction 
Denoising Diffusion Probabilistic Models utilize are based on a parameterized markov chain which makes it possible to sample from a prior distribution when knowing what the posterior was. Intuitively we add noise to a sample and learn to predict this noise. Predicting the proper noise then allows us to compute backwards from a randomly initialized to a clean (synthetic) sample.

## Training and Inference
Using the [training script](./cluster_ddpm.py) we utilize the DDPM Pytorch implementation provided by [lucidrains](https://github.com/lucidrains/denoising-diffusion-pytorch). We take raw event waveforms, compress them into latent 256 by 256 representations(see [utils](../../utils/README.md) for more information) and train our model to generate those latents. The trained model is then save and can be loaded in, see [inference notebook](./inference_test.ipynb).
The results so far suggest that synthesizing latent samples works but not in a good quality yet. Decompressing synthetic latents results in bird-like sounds and you can confidently say it learned proper. Fine-tuning and parameter exploration are necessary for better results.
