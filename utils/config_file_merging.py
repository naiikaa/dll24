from omegaconf import OmegaConf
# can be used for reducing number of config files in the future for personal usage
files = [
    "./configs_birdset/datamodule/HSN.yaml",
    "./configs_birdset/module/multilabel.yaml",
    "./configs_birdset/module/network/efficientnet.yaml",
    "./configs_birdset/datamodule/transforms/bird_default_multilabel.yaml",
    "./configs_birdset/trainer/single_gpu.yaml",
    "./configs_birdset/callbacks/default.yaml",  # callbacks
    "./configs_birdset/paths/default.yaml",
    "./configs_birdset/hydra/default.yaml"
]

base_cfg = OmegaConf.load(
    "../notebooks/reproduce_training_ekaterina/configs_birdset/experiment/birdset_neurips24/HSN/DT/efficientnet.yaml")
for file in files:
    new_cfg = OmegaConf.load(file)
    base_cfg = OmegaConf.merge(base_cfg, new_cfg)
    print(f"Loaded {file}:")

print(OmegaConf.to_yaml(base_cfg))
output_file = '../notebooks/reproduce_training_ekaterina/configs_local/merged_config.yaml'
OmegaConf.save(base_cfg, output_file)