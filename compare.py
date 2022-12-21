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


def almost_average_model(model_1, model_2):
    weights_1 = model_1.state_dict()
    weights_2 = model_2.state_dict()
    model_av = copy.deepcopy(model_1)
    weights_av = model_av.state_dict()
    for key in weights_1:
        weights_av[key] = (weights_1[key] + weights_2[key]) / 2
    last_layer_weights_1 = weights_1['decoder_transformer.layers.2.weight']
    last_layer_bias_1 = weights_1['decoder_transformer.layers.2.weight']
    last_layer_weights_2 = weights_2['decoder_transformer.layers.2.weight']
    last_layer_bias_2 = weights_2['decoder_transformer.layers.2.weight']
    last_layer_weights_comb, last_layer_bias_comb = combined_last_layer_weights(last_layer_weights_1, last_layer_bias_1, last_layer_weights_2, last_layer_bias_2)
    weights_av['decoder_transformer.layers.2.weight'] = last_layer_weights_comb
    weights_av['decoder_transformer.layers.2.bias'] = last_layer_bias_comb
    model_av.load_state_dict(weights_av)
    return model_av


def combined_last_layer_weights(last_layer_weights_1, last_layer_weights_2, last_layer_bias_1, last_layer_bias_2):
    pass
    #return last_layer_weights_comb, last_layer_bias_comb
