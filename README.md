# Deep Learning Lab 2024 - Team Generative Models 
The foundation for this project is the BirdSet dataset provide by the [**DBD-research-group**](https://github.com/DBD-research-group).

sources:
[huggingface](https://huggingface.co/datasets/DBD-research-group/BirdSet) | 
[github](https://github.com/DBD-research-group/BirdSet) | 
[paper](https://arxiv.org/abs/2403.10380)

## Introduction
This project repo is about the question whether it is possible to synthesize avian audio (bird sounds). This questions was investigated by us from different angles:
- raw waveform synthesis via models which are capable of processing time series data (1D approach)
- compression of raw waveforms into different representations (latent samples)
- latent representation synthesis via models which are capable of processing image-like data (2D approach)

![image](https://github.com/user-attachments/assets/6ce57ac7-b70f-4d1f-bb48-0ac2d5ad986f)


You will find many different model architectures and experiments which investigate one of those presented tasks.

## Content
### [analysis](analysis/readMe.md)
Contains in-depth research and comparative studies focused on generative models used for audio synthesis, particularly bird sound generation.
### [models](models/README.md)
Model implementations, scripts for training/inference and experiments with different model architectures.
### [notebooks](notebooks/README.md)
Leftovers of the exploitive phase, lots of notebooks and tests.
### [utils](utils/README.md)
Extracting events from BirdSet, compression into latent samples and custom dataset classes.

![image](https://github.com/user-attachments/assets/07e70605-da8e-4a7b-9789-05bda16fc2c7)


## Members:
- Ekaterina S.
- Iman G.
- Nikita P.
