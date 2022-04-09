import os
from torch.cuda import is_available
from torch import device
from .logs import log
from .loss_func import get_loss_func
from .bool import get_bool

class sys_config():
    def __init__(self, args, cfg, train=True):
        self.train = train
        self.model = args
        self.model_name = cfg["model_name"]
        self.model_mode = cfg["model_mode"]
        self.color_channel = cfg["color_channel"]
        self.Epoch = cfg["Epoch"]
        self.scale_pos = "init"
        self.device = "cuda:0"
        self.device_in_prog = None
        self.model_args = None
        self.save_step = cfg["save_step"]
        self.scale_factor = cfg["scale_factor"]
        self.dataset = cfg["dataset"]
        self.patch_size = cfg["patch_size"]
        self.__set_scale()
        self.optim_args = None
        self.parallel = False
        if "device" in cfg:
            self.device = cfg["device"]
        if "model_args" in cfg:
            self.model_args = cfg["model_args"]
        if "scale_position" in cfg:
            self.scale_pos = cfg["scale_position"]
        if self.train:
            self.batch_size = cfg["batch_size"]
            self.weight_init = cfg["weight_init"]
            self.loss_function = cfg["loss_function"]
            self.optim = cfg["optimizer"]
            self.loss_args = None
            if "loss_args" in cfg:
                self.loss_args = cfg["loss_args"]
            if "optim_args" in cfg:
                self.optim_args = cfg["optim_args"]
        self.__check_cuda()
        self.device_in_prog = device(self.device_in_prog)
    def __check_cuda(self):
        if "cuda" in self.device:
            cuda_idx = self.device.split(':')
            cuda_idx = cuda_idx[1].replace(' ', '')
            if self.train:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx[0]
            if not is_available():
                print(self.device + " is not useable, now try to use cpu!")
                self.device = "cpu"
                self.device_in_prog = 'cpu'
            else:
                if not cuda_idx == "0":
                    cuda_idx_int = []
                    if ',' in cuda_idx:
                        idx = cuda_idx.split(',')
                        for name in idx:
                            cuda_idx_int.append(int(name))
                    else:
                        cuda_idx_int.append(int(cuda_idx))
                    if len(cuda_idx_int) > 1:
                        if self.train:
                            self.parallel = True
                            self.device_in_prog = "cuda"
                        else:
                            self.device_in_prog = "cuda:0"
                    else:
                        self.device_in_prog = "cuda:0"
                else:
                    self.device_in_prog = "cuda:0"
        else:
            self.device = "cpu"
            self.device_in_prog = "cpu"
    def __set_scale(self):
        if not isinstance(self.scale_factor, list):
            scale_list = []
            scale_list.append(self.scale_factor)
            self.scale_factor = scale_list
    def show(self):
        log("-------------This is system config--------------", self.model_name)
        log("Model: " + self.model, self.model_name)
        log("Dataset: " + self.dataset, self.model_name)
        log("Upsample Position: " + self.model_mode, self.model_name)
        log("Color Channel: " + self.color_channel, self.model_name)
        log("Batch Size: " + str(self.batch_size), self.model_name)
        log("Patch Size: " + str(self.patch_size), self.model_name)
        log("Training Epoch: " + str(self.Epoch), self.model_name)
        log("Training Device:" + str(self.device), self.model_name)
        log("Training Scale: " + str(self.scale_factor), self.model_name)
        log("Trained Model Save Step: " + str(self.save_step), self.model_name)
        log("Weight Init: " + self.weight_init, self.model_name)
        log("Loss Function: " + self.loss_function, self.model_name)
        log("Optimizer: " + self.optim, self.model_name)
        log("Position of Upsample Method in Model: " + self.scale_pos, self.model_name)
        if self.model_args != None:
            log("Model args: " + str(self.model_args), self.model_name)
        if self.loss_args != None:
            log("Loss function args: " + str(self.loss_args), self.model_name)
        if self.optim_args != None:
            log("Optimizer args: " + str(self.optim_args), self.model_name)
    def get_loss(self):
        if self.loss_args == None:
            return get_loss_func(self.loss_function)
        else:
            return get_loss_func(self.loss_function, self.loss_args)
    def set_test_config(self, args, test_cfg):
        self.scale_factor = list(dict.fromkeys(self.scale_factor))
        self.test_color_channel = test_cfg["color_channel"]
        self.test_all = False
        self.shave = 0
        self.shave_is_scale = False
        self.patch = None
        self.indicators = ['PSNR', 'SSIM']
        if "shave" in test_cfg:
            if test_cfg["shave"] == "scale":
                self.shave_is_scale = True
            else:
                self.shave = test_cfg["shave"]
        if "patch" in test_cfg:
            if test_cfg["patch"] == 0:
                self.patch = None
            else:
                self.patch = test_cfg["patch"]
        if "indicators" in test_cfg:
            self.indicators = test_cfg["indicators"]
        if args.all:
            self.drew = get_bool(test_cfg["drew_pic"])
            self.test_all = True
        else:
            self.drew = False
        self.test_dataset = test_cfg["test_dataset"]
        self.test_file = ""
        if args.once != None:
            self.test_file = args.once
        if args.dataset != None:
            self.test_dataset = [args.dataset]