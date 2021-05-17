import os
import models
import utils
import torch
import json
import argparse

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("cfg_file", help = "config file", type = str)
    parser.add_argument("--once", default = None, type = str, help="Your model parameter file path")
    parser.add_argument("--all", action = "store_true")
    parser.add_argument("--dataset", default = None, type = str)
    args = parser.parse_args()
    return args

def load_json(args):
    cfg_file = os.path.join("./config/", args + ".json")
    with open(cfg_file, "r") as f:
        config = json.load(f)
    return config

def test():
    args = parse_args()
    config = load_json(args.cfg_file)
    hyperpara = utils.sys_config(args.cfg_file, config["system"], False)
    hyperpara.set_test_config(args, config["test"])
    tester = utils.Tester(hyperpara, config["dataloader"])
    tester.test()

if __name__ == "__main__":
    test()