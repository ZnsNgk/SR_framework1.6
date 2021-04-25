import os
import models
import utils
import torch
import json
import argparse


torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("cfg_file", help = "config file", type = str)
    parser.add_argument("--breakpoint", default = None, type = str,
                        help="If your model crashes when training, you can set breakpoints to resume training")
    args = parser.parse_args()
    return args

def load_json(args):
    cfg_file = os.path.join("./config/", args + ".json")
    with open(cfg_file, "r") as f:
        config = json.load(f)
    return config

def train():
    args = parse_args()
    config = load_json(args.cfg_file)
    hyperpara = utils.sys_config(args.cfg_file, config["system"], True)
    data = utils.Data(hyperpara, config["dataloader"], True)
    trainer = utils.Trainer(hyperpara, data, config["learning_rate"], args.breakpoint)
    trainer.train()

if __name__ == "__main__":
    train()