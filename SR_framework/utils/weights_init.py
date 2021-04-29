import torch.nn as nn
import math
from .logs import log

def none_init(model, name):
    pass

def xavier_init(model, name):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
    log("Xavier init success!", name, True)

def normalization_init(model, name):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
            nn.init.zeros_(m.bias.data)
    log("Weights normalization success!", name, True)


init_list = {
    "None": none_init,
    "Xavier": xavier_init,
    "Normal": normalization_init
}

def init_weights(name, model, model_name):
    init_list[name](model, model_name)