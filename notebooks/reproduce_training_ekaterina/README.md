Reproducing Training Steps from:
[https://github.com/DBD-research-group/BirdSet/blob/main/README.md](https://github.com/DBD-research-group/BirdSet/blob/main/README.md)

`./configs_local` and `./datamodule` are slightly adjusted modules used in BirdSet to work in this environment due to some runtime errors with configurations from the imported original package.

### TODOs:
- Update code for sound data transformation (causes errors when trying to fit data using the original configuration).
- Refactor `reproducing_training.py` into a notebook.
- Add a description for installation and importing of original configurations (not pushed to Git) and change imports to only include modified configs. The rest must be used directly from the BirdSet package.
