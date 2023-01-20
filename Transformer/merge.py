import copy
import torch
import numpy as np
from scipy import optimize


def average_model(model_1, model_2):
    weights_1 = model_1.state_dict()
    weights_2 = model_2.state_dict()
    model_av = copy.deepcopy(model_1)
    weights_av = model_av.state_dict()
    for key in weights_1:
        weights_av[key] = (weights_1[key] + weights_2[key]) / 2
    model_av.load_state_dict(weights_av)
    return model_av


def perturbed_embedding(model_1, model_2):
    weights_1 = model_1.state_dict()
    weights_2 = model_2.state_dict()
    model_av = copy.deepcopy(model_1)
    weights_av = model_av.state_dict()

    # For the weights we're not permuting, we just average them
    for key in weights_1:
        weights_av[key] = (weights_1[key] + weights_2[key]) / 2

    # what would happen if we just copied the embedding from one model to the other?

    # gets the weights used to determine the permutation
    embedding_1 = weights_1['encoder.weight']
    embedding_2 = weights_2['encoder.weight']
    decoder_1 = weights_1['decoder.weight']
    decoder_2 = weights_2['decoder.weight']

    embedding_permutation = get_best_permutation(embedding_1, embedding_2)
    decoder_permutation = get_best_permutation(decoder_1, decoder_2)

    print('permuted nodes in embedding', np.count_nonzero(embedding_permutation - np.arange(embedding_permutation.shape[0])))
    print('permuted nodes in decoder', np.count_nonzero(decoder_permutation - np.arange(decoder_permutation.shape[0])))

    permuted_embedding = permute_weights_out(embedding_permutation, embedding_2)
    permuted_decoder = permute_weights_out(decoder_permutation, decoder_2)

    av_embedding = (embedding_1 + permuted_embedding) / 2
    av_decoder = (decoder_1 + permuted_decoder) / 2

    # loads the permuted weights into the model
    weights_av['encoder.weight'] = av_embedding
    weights_av['decoder.weight'] = av_decoder
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
    permuted_last_layer_weights_2_in = permute_weights_in(permutation, last_layer_weights_2_in)
    permuted_last_layer_bias_2_in = permute_weights_in(permutation, last_layer_bias_2_in)
    permuted_last_layer_weights_2_out = permute_weights_out(permutation, last_layer_weights_2_out)

    # You forgot to average the permuted stuff.
    av_weights_in = (weights_1['transformer_encoder.layers.7.linear1.weight'] + permuted_last_layer_weights_2_in) / 2
    av_bias_in = (weights_1['transformer_encoder.layers.7.linear1.bias'] + permuted_last_layer_bias_2_in) / 2
    av_weights_out = (weights_1['transformer_encoder.layers.7.linear2.weight'] + permuted_last_layer_weights_2_out) / 2

    # loads the permuted weights into the model
    weights_av['transformer_encoder.layers.7.linear1.weight'] = av_weights_in
    weights_av['transformer_encoder.layers.7.linear1.bias'] = av_bias_in
    weights_av['transformer_encoder.layers.7.linear2.weight'] = av_weights_out
    model_av.load_state_dict(weights_av)
    return model_av


def permute_weights_in(permutation, weights_in):
    return weights_in[permutation]


def permute_weights_out(permutation, weights_out):
    return weights_out[:, permutation]


"""
Calculates the cost (difference in weights) for each node individually being mapped to another node. 
All of these costs are put into a matrix. 
Then uses linear sum assignment solver to find the best permutation
"""
def get_best_permutation(true_weights, merge_weights, true_bias=None, merge_bias=None):
    cost_matrix = get_cost_matrix(true_weights, merge_weights, true_bias, merge_bias)
    _, col_idx = optimize.linear_sum_assignment(cost_matrix)
    return col_idx


def get_cost_matrix(true_weights, merge_weights, true_bias=None, merge_bias=None):
    n = true_weights.shape[1]
    cost_matrix = np.zeros((n, n))
    for true_node in range(n):
        for merge_node in range(n):
            cost_matrix[true_node, merge_node] = cost(true_weights[:, true_node], merge_weights[:, merge_node], true_bias, merge_bias)
    return cost_matrix


def cost(true_weights, merge_weights, true_bias=None, merge_bias=None):
    if true_bias is None:
        bias_sum = 0
    else:
        bias_sum = torch.sum(torch.abs(true_bias - merge_bias))
    return torch.sum(torch.abs(true_weights - merge_weights)) + bias_sum
