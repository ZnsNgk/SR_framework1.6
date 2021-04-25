

model_list = {

}

def get_model(model_name, **kwargs):
    return model_list[model_name](**kwargs)