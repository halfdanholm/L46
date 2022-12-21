import copy


def average_model(model_1, model_2):
    weights_1 = model_1.state_dict()
    weights_2 = model_2.state_dict()
    model_av = copy.deepcopy(model_1)
    weights_av = model_av.state_dict()
    for key in weights_1:
        weights_av[key] = (weights_1[key] + weights_2[key]) / 2
    model_av.load_state_dict(weights_av)
    return model_av
