import os, sys, time
import numpy as np
import torch
import argparse
from common.utils import *
from data_prep import data_handler
import wandb
os.environ["WANDB_SILENT"] = "true"
parser = argparse.ArgumentParser(description='Process hyper-parameters')
parser.add_argument('--yaml', type=str, default='config/hparams.yaml')
args = parser.parse_args()

def main():

    cfg = read_yaml(args.yaml)
    if cfg.common.decode_only:
        loaders = data_handler.collect(cfg, shuffle=False)
    else:
        loaders = data_handler.collect(cfg, shuffle=True)
    model, cfg = get_model(cfg)
    trainer = get_trainer(cfg)
    operate = trainer.Operate(cfg)
    if cfg.common.decode_only:
        output = operate.trainer(model, loaders, decode=True)
    else:
        operate.trainer(model, loaders)


if __name__ == '__main__':
    main()
