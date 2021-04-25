import torch.nn as nn
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
            nn.init.normal_(m.weight)
    log("Weights normalization success!", name, True)


init_list = {
    "None": none_init,
    "Xavier": xavier_init,
    "Normal": normalization_init
}

def init_weights(name, model, model_name):
    init_list[name](model, model_name)