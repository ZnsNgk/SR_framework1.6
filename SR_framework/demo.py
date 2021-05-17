import os
import models
import utils
import cv2
import torch
import json
import numpy
import argparse

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("cfg_file", help = "config file", type = str)
    parser.add_argument("--file", default = None, type = str, help="Your model parameter file path")
    parser.add_argument("--input", action = "store_true")
    parser.add_argument("--dataset", default = None, type = str)
    args = parser.parse_args()
    return args

def load_json(args):
    cfg_file = os.path.join("./config/", args + ".json")
    with open(cfg_file, "r") as f:
        config = json.load(f)
    return config

class Demo():
    def __init__(self, args, config):
        self.model = args.cfg_file
        self.device = config["system"]["device"]
        self.model_name = config["system"]["model_name"]
        self.model_mode = config["system"]["model_mode"]
        self.scale_pos = config["system"]["scale_position"]
        self.is_normal = utils.get_bool(config["dataloader"]["normalize"])
        self.is_Y = False
        if config["system"]["color_channel"] == "RGB":
            self.is_Y = False
        elif config["system"]["color_channel"] == "Y":
            self.is_Y = True
        self.model_args = None
        self.file = args.file
        self.file_path = './trained_model/' + self.model_name + '/' + self.file
        self.save_path = './demo_output/' + self.model_name + '/'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.__check_cuda()
        if "model_args" in config["system"]:
            self.model_args = config["system"]["model_args"]
        if "pkl" in self.file:
            self.is_pkl = True
            self.scale = int(self.file.replace(".pkl", "").replace("x", ""))
        else:
            self.is_pkl = False
            state_list = self.file.split('_')
            self.scale = int(state_list[1].replace('x', ''))
        if args.input:
            self.is_input = True
            self.dataset = './demo_input'
        else:
            self.is_input = False
            self.dataset = './data/test/' + args.dataset
    def __check_cuda(self):
        if "cuda" in self.device:
            if not torch.cuda.is_available():
                print("Cuda is not useable, now try to use cpu!")
                self.device = "cpu"
        self.device = torch.device(self.device)
    def __get_model(self):
        if self.model_mode == "post":
            if self.scale_pos == "init":
                if self.model_args == None:
                    return models.get_model(self.model, scale=self.scale)
                else:
                    return models.get_model(self.model, scale=self.scale, **self.model_args)
            elif self.scale_pos == "forward":
                if self.model_args == None:
                    return models.get_model(self.model)
                else:
                    return models.get_model(self.model, **self.model_args)
        elif self.model_mode == "pre":
            if self.model_args == None:
                return models.get_model(self.model)
            else:
                return models.get_model(self.model, **self.model_args)
        else:
            raise NameError("ERROR model_mode!")
    def __scale_pos_is_forward(self, net, loader, is_normal):
        for _, data in enumerate(loader):
            img, name = data
            sr = net(img.to(self.device), self.scale)
            sr = sr.permute(0, 2, 3, 1).squeeze(0)
            if is_normal:
                sr = sr * 255.
            sr = numpy.array(sr.cpu())
            cv2.imwrite(self.save_path+name[0], sr)
            print(name[0] + " Success!")
    def __pre_or_init(self, net, loader, is_normal):
        for _, data in enumerate(loader):
            img, name = data
            sr = net(img.to(self.device))
            sr = sr.permute(0, 2, 3, 1).squeeze(0)
            if is_normal:
                sr = sr * 255.
            sr = numpy.array(sr.cpu())
            cv2.imwrite(self.save_path+name[0], sr)
            print(name[0] + " Success!")
    def run_demo(self):
        if self.is_pkl:
            net = torch.load(self.file_path, map_location=self.device)
        else:
            net = self.__get_model()
            para = torch.load(self.file_path, map_location=self.device)
            net.load_state_dict(para)
        net.eval()
        net.to(self.device)
        loader = utils.get_demo_loader(self.dataset, self.scale, self.is_normal, self.is_input, self.is_Y)
        if self.model_mode == "post":
            if self.scale_pos == "init":
                self.__pre_or_init(net, loader, self.is_normal)
            elif self.scale_pos == "forward":
                self.__scale_pos_is_forward(net, loader, self.is_normal)
            else:
                raise NameError("WRONG MODEL SCALE POSITION!")
        elif self.model_mode == "pre":
            self.__pre_or_init(net, loader, self.is_normal)
        else:
            raise NameError("WRONG MODEL MODE!")

def demo():
    args = parse_args()
    config = load_json(args.cfg_file)
    demo = Demo(args, config)
    demo.run_demo()

if __name__ == "__main__":
    demo()