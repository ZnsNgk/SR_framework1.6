'''
Import your models here and join them into model_list
'''

model_list = {

}

def get_model(model_name, **kwargs):
    return model_list[model_name](**kwargs)