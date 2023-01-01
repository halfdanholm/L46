import copy

import numpy as np


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


def find_hungarian_mapping(batch_weights, layer_index, batch_frequencies, sigma_layers,
                                 sigma0_layers, gamma_layers, it,
                                 model_meta_data,
                                 model_layer_type,
                                 n_layers,
                                 matching_shapes,
                                 args):
    """
    We implement a layer-wise matching here:
    """
    if type(sigma_layers) is not list:
        sigma_layers = (n_layers - 1) * [sigma_layers]
    if type(sigma0_layers) is not list:
        sigma0_layers = (n_layers - 1) * [sigma0_layers]
    if type(gamma_layers) is not list:
        gamma_layers = (n_layers - 1) * [gamma_layers]

    last_layer_const = []
    total_freq = sum(batch_frequencies)
    for f in batch_frequencies:
        last_layer_const.append(f / total_freq)

    # J: number of workers
    J = len(batch_weights)
    # init_num_kernel: the number of conv filters in the first conv layer
    init_num_kernel = batch_weights[0][0].shape[0]

    # for saving (#channel * k * k)
    init_channel_kernel_dims = []
    for bw in batch_weights[0]:
        if len(bw.shape) > 1:
            init_channel_kernel_dims.append(bw.shape[1])
    logger.info("init_channel_kernel_dims: {}".format(init_channel_kernel_dims))

    sigma_bias_layers = sigma_layers
    sigma0_bias_layers = sigma0_layers
    mu0 = 0.
    mu0_bias = 0.1
    assignment_c = [None for j in range(J)]
    L_next = None

    sigma = sigma_layers[layer_index - 1]
    sigma_bias = sigma_bias_layers[layer_index - 1]
    gamma = gamma_layers[layer_index - 1]
    sigma0 = sigma0_layers[layer_index - 1]
    sigma0_bias = sigma0_bias_layers[layer_index - 1]

    # our assumption is that this branch will consistently handle the last fc layers
    layer_type = model_layer_type[2 * layer_index - 2]
    prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]
    first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and (
                'conv' in prev_layer_type or 'features' in layer_type))

    # we switch to ignore the last layer here:
    if first_fc_identifier:
        weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T,
                                   batch_weights[j][2 * layer_index - 1].reshape(-1, 1))) for j in range(J)]
    else:
        weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T,
                                   batch_weights[j][2 * layer_index - 1].reshape(-1, 1))) for j in range(J)]

    sigma_inv_prior = np.array([1 / sigma0_bias] + (weights_bias[0].shape[1] - 1) * [1 / sigma0])
    mean_prior = np.array([mu0_bias] + (weights_bias[0].shape[1] - 1) * [mu0])

    assignment_c, global_weights_c, global_sigmas_c = match_layer(weights_bias, sigma_inv_layer, mean_prior,
                                                                  sigma_inv_prior, gamma, it)

    L_next = global_weights_c.shape[0]

    softmax_bias, softmax_inv_sigma = process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)

    layer_type = model_layer_type[2 * layer_index - 2]
    prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]
    # first_fc_identifier = ('fc' in layer_type and 'conv' in prev_layer_type)
    first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and (
                'conv' in prev_layer_type or 'features' in layer_type))
    layer_type = model_layer_type[2 * layer_index - 2]
    gwc_shape = global_weights_c.shape
    if "conv" in layer_type or 'features' in layer_type:
        global_weights_out = [global_weights_c[:, 0:gwc_shape[1] - 1], global_weights_c[:, gwc_shape[1] - 1]]
        global_inv_sigmas_out = [global_sigmas_c[:, 0:gwc_shape[1] - 1], global_sigmas_c[:, gwc_shape[1] - 1]]
    elif "fc" in layer_type or 'classifier' in layer_type:
        global_weights_out = [global_weights_c[:, 0:gwc_shape[1] - 1].T, global_weights_c[:, gwc_shape[1] - 1]]
        global_inv_sigmas_out = [global_sigmas_c[:, 0:gwc_shape[1] - 1].T, global_sigmas_c[:, gwc_shape[1] - 1]]

    logger.info("#### Branch B, Layer index: {}, Global weights out shapes: {}".format(layer_index,
                                                                                       [gwo.shape for gwo in
                                                                                        global_weights_out]))

    logger.info("global inv sigma out shape: {}".format([giso.shape for giso in global_inv_sigmas_out]))
    map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
    return map_out, assignment_c, L_next


def match_layer(weights_bias, sigma_inv_layer, mean_prior, sigma_inv_prior, gamma, it):
    J = len(weights_bias)

    group_order = sorted(range(J), key=lambda x: -weights_bias[x].shape[0])

    batch_weights_norm = [w * s for w, s in zip(weights_bias, sigma_inv_layer)]
    prior_mean_norm = mean_prior * sigma_inv_prior

    global_weights = prior_mean_norm + batch_weights_norm[group_order[0]]
    global_sigmas = np.outer(np.ones(global_weights.shape[0]), sigma_inv_prior + sigma_inv_layer[group_order[0]])

    popularity_counts = [1] * global_weights.shape[0]

    assignment = [[] for _ in range(J)]

    assignment[group_order[0]] = list(range(global_weights.shape[0]))

    # Why are we iterating over the group order?
    ## Initialize
    for j in group_order[1:]:
        global_weights, global_sigmas, popularity_counts, assignment_j = matching_upd_j(batch_weights_norm[j],
                                                                                        global_weights,
                                                                                        sigma_inv_layer[j],
                                                                                        global_sigmas, prior_mean_norm,
                                                                                        sigma_inv_prior,
                                                                                        popularity_counts, gamma, J)
        assignment[j] = assignment_j

    ## Iterate over groups
    for iteration in range(it):
        random_order = np.random.permutation(J)
        for j in random_order:  # random_order:
            to_delete = []
            ## Remove j
            Lj = len(assignment[j])
            for l, i in sorted(zip(range(Lj), assignment[j]), key=lambda x: -x[1]):
                popularity_counts[i] -= 1
                if popularity_counts[i] == 0:
                    del popularity_counts[i]
                    to_delete.append(i)
                    for j_clean in range(J):
                        for idx, l_ind in enumerate(assignment[j_clean]):
                            if i < l_ind and j_clean != j:
                                assignment[j_clean][idx] -= 1
                            elif i == l_ind and j_clean != j:
                                logger.info('Warning - weird unmatching')
                else:
                    global_weights[i] = global_weights[i] - batch_weights_norm[j][l]
                    global_sigmas[i] -= sigma_inv_layer[j]

            global_weights = np.delete(global_weights, to_delete, axis=0)
            global_sigmas = np.delete(global_sigmas, to_delete, axis=0)

            ## Match j
            global_weights, global_sigmas, popularity_counts, assignment_j = matching_upd_j(batch_weights_norm[j],
                                                                                            global_weights,
                                                                                            sigma_inv_layer[j],
                                                                                            global_sigmas,
                                                                                            prior_mean_norm,
                                                                                            sigma_inv_prior,
                                                                                            popularity_counts, gamma, J)
            assignment[j] = assignment_j

    return assignment, global_weights, global_sigmas


def matching_upd_j(weights_j, global_weights, sigma_inv_j, global_sigmas, prior_mean_norm, prior_inv_sigma,
                   popularity_counts, gamma, J):

    L = global_weights.shape[0]

    full_cost = compute_cost(global_weights.astype(np.float32), weights_j.astype(np.float32), global_sigmas.astype(np.float32), sigma_inv_j.astype(np.float32), prior_mean_norm.astype(np.float32), prior_inv_sigma.astype(np.float32),
                             popularity_counts, gamma, J)

    # please note that this can not run on non-Linux systems
    row_ind, col_ind = solve_dense(-full_cost)
    assignment_j = []

    new_L = L

    for l, i in zip(row_ind, col_ind):
        if i < L:
            popularity_counts[i] += 1
            assignment_j.append(i)
            global_weights[i] += weights_j[l]
            global_sigmas[i] += sigma_inv_j
        else:  # new neuron
            popularity_counts += [1]
            assignment_j.append(new_L)
            new_L += 1
            global_weights = np.vstack((global_weights, prior_mean_norm + weights_j[l]))
            global_sigmas = np.vstack((global_sigmas, prior_inv_sigma + sigma_inv_j))

    return global_weights, global_sigmas, popularity_counts, assignment_j


# Cost should be what?
def compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                 popularity_counts, gamma, J):

    Lj = weights_j.shape[0]
    counts = np.minimum(np.array(popularity_counts, dtype=np.float32), 10)

    sij_p_gs = sigma_inv_j + global_sigmas
    red_term = (global_weights ** 2 / global_sigmas).sum(axis=1)
    param_cost = np.array([row_param_cost_simplified(global_weights, weights_j[l], sij_p_gs, red_term) for l in range(Lj)], dtype=np.float32)

    param_cost += np.log(counts / (J - counts))

    L = global_weights.shape[0]
    max_added = min(Lj, max(700 - L, 1))
    nonparam_cost = np.outer((((weights_j + prior_mean_norm) ** 2 / (prior_inv_sigma + sigma_inv_j)).sum(axis=1) - (
                prior_mean_norm ** 2 / prior_inv_sigma).sum()), np.ones(max_added, dtype=np.float32))
    cost_pois = 2 * np.log(np.arange(1, max_added + 1))
    nonparam_cost -= cost_pois
    nonparam_cost += 2 * np.log(gamma / J)

    full_cost = np.hstack((param_cost, nonparam_cost)).astype(np.float32)
    return full_cost


def row_param_cost_simplified(global_weights, weights_j_l, sij_p_gs, red_term):
    match_norms = ((weights_j_l + global_weights) ** 2 / sij_p_gs).sum(axis=1) - red_term
    return match_norms


def combined_last_layer_weights(last_layer_weights_1, last_layer_weights_2, last_layer_bias_1, last_layer_bias_2):
    pass
    #return last_layer_weights_comb, last_layer_bias_comb
