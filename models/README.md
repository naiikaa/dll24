# Models
This part contains all model architectures that were used for our generative experiments each folder contains notebooks and scripts where we tried to utilize the corresponding model architecture to generate raw waveforms or latent representations of those waveforms.
## List of tested models
### GAN architectures
- [DCGAN](GAN/DCGAN/DCGAN.py) for generating 2D latents
- [WGAN](GAN/WGAN/WGAN.py) for generating 2D latents
- [WGAN-GP](GAN/WGAN-GP/WGAN-GP.py) for generating 2D latents
### DDPM
- [DDPM](DDPM/cluster_ddpm.py) for generating 2D latents
### WAVENET
- [WAVENET_V1](WAVENET/wavenet_v1/WAVENET_TRAIN.ipynb) for generating .wav files 
- [WAVENET_V2](WAVENET/wavenet_v2/WAVENET_TRAIN.ipynb) for generating .wav files
- [waveNet_igz_V03.py](WAVENET/waveNet_igz/waveNet_igz_V03.py) for generating '`wav' file
### VQVAE2
- [ECOGEN_V](VQVAE2/ECOGEN_V/VQVAE2_from_ECOGEN.ipynb) for generating .wav files
- [VQVAE2_LATENT](VQVAE2/VQVAE2_LATENT/VQVAE2_LATENT.ipynb) for generating 2D latents 
- [VQVAE2_SPEC](VQVAE2/VQVAE2_SPEC/VQVAE2.ipynb) for generating 2D spectrograms
### WAVEGLOW
- [WAVEGLOW](WAVEGLOW/WAVEGLOW.ipynb) for generating 2D spectrograms

### HiFiGan
- [hifiGan_quick_check_V.01.py](HiFiGan/hifiGan_quick_check_V.01.py) for generating `.wav` files
