import copy
import torch
import numpy as np
import scipy as sp


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

    # For the weights we're not permuting, we just average them
    for key in weights_1:
        weights_av[key] = (weights_1[key] + weights_2[key]) / 2

    # gets the weights used to determine the permutation
    last_layer_weights_1_out = weights_1['transformer_encoder.layers.7.linear2.weight']
    last_layer_bias_1_in = weights_1['transformer_encoder.layers.7.linear1.bias']
    last_layer_weights_2_out = weights_2['transformer_encoder.layers.7.linear2.weight']
    last_layer_bias_2_in = weights_2['transformer_encoder.layers.7.linear1.bias']
    permutation = get_best_permutation(last_layer_weights_1_out, last_layer_weights_2_out, last_layer_bias_1_in,
                                       last_layer_bias_2_in)

    print('permuted nodes', np.count_nonzero(permutation - np.arange(permutation.shape[0])))

    # permutes the weights that need to be permuted
    last_layer_weights_2_in = weights_2['transformer_encoder.layers.7.linear1.weight']
    permuted_weights_in = permute_weights_in(permutation, [last_layer_weights_2_in, last_layer_bias_2_in])
    permuted_weights_out = permute_weights_out(permutation, [last_layer_weights_2_out])

    # loads the permuted weights into the model
    weights_av['transformer_encoder.layers.7.linear2.weight'] = permuted_weights_out[0]
    weights_av['transformer_encoder.layers.7.linear1.bias'] = permuted_weights_in[1]
    weights_av['transformer_encoder.layers.7.linear1.weight'] = permuted_weights_in[0]
    model_av.load_state_dict(weights_av)
    return model_av


def permute_weights_in(permutation, weights_in):
    return [w[permutation] for w in weights_in]


def permute_weights_out(permutation, weights_out):
    return [w[:, permutation] for w in weights_out]


"""
Calculates the cost (difference in weights) for each node individually being mapped to another node. 
All of these costs are put into a matrix. 
Then uses linear sum assignment solver to find the best permutation
"""
def get_best_permutation(true_weights, merge_weights, true_bias, merge_bias):
    cost_matrix = get_cost_matrix(true_weights, merge_weights, true_bias, merge_bias)
    _, col_idx = sp.optimize.linear_sum_assignment(cost_matrix)
    return col_idx


def get_cost_matrix(true_weights, merge_weights, true_bias, merge_bias):
    n = true_weights.shape[1]
    cost_matrix = np.zeros((n, n))
    for true_node in range(n):
        for merge_node in range(n):
            cost_matrix[true_node, merge_node] = cost(true_weights[:, true_node], merge_weights[:, merge_node], true_bias[true_node], merge_bias[merge_node])
    return cost_matrix


def cost(true_weights, merge_weights, true_bias, merge_bias):
    return weight_difference(
        true_weights, merge_weights, true_bias, merge_bias
    )


def weight_difference(true_weights, merge_weights, true_bias, merge_bias):
    return torch.sum(torch.abs(true_weights - merge_weights)) + torch.sum(torch.abs(true_bias - merge_bias))
