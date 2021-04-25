import torch.optim as optim

optim_list ={
    "SGD": optim.SGD,
    "Adam": optim.Adam
}

def get_optimizer(name, para, learning_rate, **kwargs):
    return optim_list[name](para, lr=learning_rate, **kwargs)

class none_scheduler():
    def __init__(self, optim, **kwargs):
        self.args = kwargs
        self.name = "None"
    def step(self):
        pass

scheduler_list = {
    "None": none_scheduler,
    "Step": optim.lr_scheduler.StepLR,
    "Exp": optim.lr_scheduler.ExponentialLR,
    "MultiStep": optim.lr_scheduler.MultiStepLR,
    "CosineAnnealing": optim.lr_scheduler.CosineAnnealingLR
}

def get_scheduler(name, optim, **kwargs):
    for p in list(kwargs.keys()):
        if not kwargs[p]:
            del kwargs[p]
    if name == "MultiStep":
        kwargs['milestones'] = kwargs['step_size']
        del kwargs['step_size']
    elif name == "CosineAnnealing":
        kwargs['T_max'] = kwargs['step_size']
        del kwargs['step_size']
    return scheduler_list[name](optim, **kwargs)