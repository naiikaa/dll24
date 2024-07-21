import os
from pathlib import Path
import json
from omegaconf import OmegaConf, open_dict
from omegaconf import DictConfig
import lightning as L
import pyrootutils
import hydra
from birdset import utils
import torch
from birdset.modules.base_module import BaseModule
from datamodule.components.augmentations import MultilabelMix

# trying to reproduce original training with custom settings / partially changed code
# to make it runnable on a local machine
def print_module_structure(start_path, indent=''):
    for item in os.listdir(start_path):
        path = os.path.join(start_path, item)
        if os.path.isdir(path):
            print(f"{indent}{item}/")
            print_module_structure(path, indent + '    ')
        elif item.endswith('.py'):
            print(f"{indent}{item}")


# TODO:
# no gpu? *megamind meme comes here*
# print(torch.cuda.is_available())

# print_module_structure('./')
relative_path = os.path.abspath('./')
os.environ['PROJECT_ROOT'] = relative_path
os.environ['HYDRA_FULL_ERROR'] = '1'
# Relative path
relative_path = './configs_birdset'
root = os.path.abspath(relative_path)
print(root)
_HYDRA_PARAMS = {
    "version_base": None,
    "config_path": str(root),
    "config_name": "train.yaml"
}


@hydra.main(**_HYDRA_PARAMS)
def train(cfg):
    print(OmegaConf.to_yaml(cfg))

    log = utils.get_pylogger(__name__)
    # API key needed - skip
    # logger = utils.instantiate_loggers(cfg.get("logger"))
    # Not adding logger due to API key error

    with open_dict(cfg):
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        datamodule.prepare_data()

        callbacks = utils.instantiate_callbacks(cfg["callbacks"])

        trainer = hydra.utils.instantiate(
            cfg.trainer, callbacks=callbacks
        )
        # .....
        cfg.module.metrics["num_labels"] = datamodule.num_classes
        cfg.module.network.model["num_classes"] = datamodule.num_classes

    model = hydra.utils.instantiate(
        cfg.module,
        num_epochs=cfg.trainer.max_epochs,
        len_trainset=datamodule.len_trainset,
        batch_size=datamodule.loaders_config.train.batch_size,
        pretrain_info=cfg.module.network.model.pretrain_info
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "trainer": trainer
    }
    utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info(f"Starting training")
        trainer.fit(
            model=model,
            datamodule=datamodule
        )

    train_metrics = trainer.callback_metrics

'''
    if cfg.get("test"):
        log.info(f"Starting testing")
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=trainer.checkpoint_callback.best_model_path
        )

    test_metrics = trainer.callback_metrics

    if cfg.get("save_state_dict"):
        log.info(f"Saving state dicts")
        utils.save_state_dicts(
            trainer=trainer,
            model=model,
            dirname=cfg.paths.output_dir,
            **cfg.extras.state_dict_saving_params
        )

    if cfg.get("dump_metrics"):
        log.info(f"Dumping final metrics locally to {cfg.paths.output_dir}")
        metric_dict = {**train_metrics, **test_metrics}
        metric_dict = [{'name': k, 'value': v.item() if hasattr(v, 'item') else v} for k, v in metric_dict.items()]
        file_path = os.path.join(cfg.paths.output_dir, "finalmetrics.json")
        with open(file_path, 'w') as json_file:
            json.dump(metric_dict, json_file)
'''

if __name__ == "__main__":
    # experiment="./configs_birdset/experiment/birdset_neurips24/HSN/DT/efficientnet.yaml"
    train()
