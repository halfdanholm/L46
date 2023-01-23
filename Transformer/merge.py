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


def almost_average_model(model_true, model_permute):
    permute_layer_by_ids(
        model_true,
        model_permute,
        [f'transformer_encoder.layers.7.linear2.weight', f'transformer_encoder.layers.7.linear1.bias'],
        [f'transformer_encoder.layers.7.linear1.weight', f'transformer_encoder.layers.7.linear1.bias'],
        [f'transformer_encoder.layers.7.linear2.weight'],
        transpose_determiner=True
    )


def permute_model(device, model_true, model_permute_in):
    model_permute = copy.deepcopy(model_permute_in)
    model_permute.to(device)
    permute_layer_by_ids(
        model_true,
        model_permute,
        ['encoder.weight'],
        [
            'transformer_encoder.layers.0.self_attn.out_proj.weight',
            'transformer_encoder.layers.0.self_attn.out_proj.bias',
            'transformer_encoder.layers.0.linear2.weight',
            'transformer_encoder.layers.0.linear2.bias'
        ],
        [
            'transformer_encoder.layers.0.self_attn.in_proj_weight',
            'encoder.weight',
            'transformer_encoder.layers.0.linear1.weight',
            'decoder.weight'
        ],
        ['pos_encoder.pe'],
        transpose_determiner=True
    )
    max_layer = 1
    for layer in range(max_layer):
        permute_queries_and_keys(model_true, model_permute, layer)
        permute_value_layer(model_true, model_permute, layer)
        permute_layer_by_ids(
            model_true,
            model_permute,
            [f'transformer_encoder.layers.{layer}.linear1.weight', f'transformer_encoder.layers.{layer}.linear1.bias'],
            [f'transformer_encoder.layers.{layer}.linear1.weight', f'transformer_encoder.layers.{layer}.linear1.bias'],
            [f'transformer_encoder.layers.{layer}.linear2.weight'],
            []
        )
        if (layer + 1) < max_layer:
            pass
            """permute_layer_by_ids(
                model_true,
                model_permute,
                [f'transformer_encoder.layers.{layer}.linear2.weight', f'transformer_encoder.layers.{layer}.linear2.bias'],
                [f'transformer_encoder.layers.{layer}.linear2.weight', f'transformer_encoder.layers.{layer}.linear2.bias'],
                [f'transformer_encoder.layers.{layer + 1}.self_attn.in_proj_weight'],
                []
            )"""
        else:
            pass
            """permute_layer_by_ids(
                model_true,
                model_permute,
                [f'transformer_encoder.layers.{max_layer - 1}.linear2.weight', f'transformer_encoder.layers.{max_layer - 1}.linear2.bias'],
                [f'transformer_encoder.layers.{max_layer - 1}.linear2.weight', f'transformer_encoder.layers.{max_layer - 1}.linear2.bias'],
                ['decoder.weight'],
                []
            )"""
    return model_permute


def permute_layer_by_ids(
        model_true,
        model_permute,
        weights_determined_ids,
        weights_affected_rows_ids,
        weights_affected_columns_ids,
        weights_affected_third_ids,
        transpose_determiner=False
):
    weights_true = model_true.state_dict()
    weights_permute = model_permute.state_dict()

    weights_determined, weights_affected_rows, weights_affected_columns, weights_affected_third = [], [], [], []
    for weights_determined_id in weights_determined_ids:
        weights_determined_true = weights_true[weights_determined_id]
        weights_determined_permute = weights_permute[weights_determined_id]
        if transpose_determiner:
            if len(weights_determined_true.shape) == 2:
                weights_determined_true = weights_determined_true.T
                weights_determined_permute = weights_determined_permute.T
            elif len(weights_determined_true.shape) == 1:
                pass
                # weights_determined_true = weights_determined_true.reshape(1, -1)
                # weights_determined_permute = weights_determined_permute.reshape(1, -1)
            else:
                raise ValueError('Unknown shape')
        weights_determined.append((weights_determined_true, weights_determined_permute))
    for weights in weights_affected_rows_ids:
        weights_affected_rows.append(weights_permute[weights])
    for weights in weights_affected_columns_ids:
        weights_affected_columns.append(weights_permute[weights])
    for weights in weights_affected_third_ids:
        weights_affected_third.append(weights_permute[weights])

    weights_affected_rows, weights_affected_columns, weights_affected_third = permute_layer(
        weights_determined,
        weights_affected_rows,
        weights_affected_columns,
        weights_affected_third
    )

    for weights_affected_rows_ids, weights_affected_rows in zip(weights_affected_rows_ids, weights_affected_rows):
        weights_permute[weights_affected_rows_ids] = weights_affected_rows
    for weights_affected_columns_ids, weights_affected_columns in zip(weights_affected_columns_ids, weights_affected_columns):
        weights_permute[weights_affected_columns_ids] = weights_affected_columns
    for weights_affected_third_ids, weights_affected_third in zip(weights_affected_third_ids, weights_affected_third):
        weights_permute[weights_affected_third_ids] = weights_affected_third

    model_permute.load_state_dict(weights_permute)


def permute_queries_and_keys(model_true, model_permute, layer):
    weights_true = model_true.state_dict()
    weights_permute = model_permute.state_dict()

    qvk_matrix_id = 'transformer_encoder.layers.' + str(layer) + '.self_attn.in_proj_weight'
    qvk_matrix_true = weights_true[qvk_matrix_id]
    qvk_matrix_permute = weights_permute[qvk_matrix_id]
    q_matrix_true = qvk_matrix_true[:qvk_matrix_true.shape[0] // 3]
    k_matrix_true = qvk_matrix_true[qvk_matrix_true.shape[0] // 3:2 * qvk_matrix_true.shape[0] // 3]
    #v_matrix_true = qvk_matrix_true[2 * qvk_matrix_true.shape[0] // 3:]
    q_matrix_permute = qvk_matrix_permute[:qvk_matrix_permute.shape[0] // 3]
    k_matrix_permute = qvk_matrix_permute[qvk_matrix_permute.shape[0] // 3:2 * qvk_matrix_permute.shape[0] // 3]
    #v_matrix_permute = qvk_matrix_permute[2 * qvk_matrix_permute.shape[0] // 3:]

    qvk_matrix_bias_id = 'transformer_encoder.layers.' + str(layer) + '.self_attn.in_proj_bias'
    qvk_matrix_bias_true = weights_true[qvk_matrix_bias_id]
    qvk_matrix_bias_permute = weights_permute[qvk_matrix_bias_id]
    q_matrix_bias_true = qvk_matrix_bias_true[:qvk_matrix_bias_true.shape[0] // 3]
    k_matrix_bias_true = qvk_matrix_bias_true[qvk_matrix_bias_true.shape[0] // 3:2 * qvk_matrix_bias_true.shape[0] // 3]
    #v_matrix_bias_true = qvk_matrix_bias_true[2 * qvk_matrix_bias_true.shape[0] // 3:]
    q_matrix_bias_permute = qvk_matrix_bias_permute[:qvk_matrix_bias_permute.shape[0] // 3]
    k_matrix_bias_permute = qvk_matrix_bias_permute[qvk_matrix_bias_permute.shape[0] // 3:2 * qvk_matrix_bias_permute.shape[0] // 3]
    #v_matrix_bias_permute = qvk_matrix_bias_permute[2 * qvk_matrix_bias_permute.shape[0] // 3:]

    affected_next_layer_id = 'transformer_encoder.layers.' + str(layer) + '.self_attn.out_proj.weight'
    affected_next_layer_permute = model_permute.state_dict()[affected_next_layer_id]

    weights_affected_rows, weights_affected_columns, _ = permute_layer(
        [
            (q_matrix_true, q_matrix_permute),
            (k_matrix_true, k_matrix_permute),
            (q_matrix_bias_true, q_matrix_bias_permute),
            (k_matrix_bias_true, k_matrix_bias_permute)
        ],
        [
            q_matrix_permute,
            k_matrix_permute,
            q_matrix_bias_permute,
            k_matrix_bias_permute
        ],
        [],
        [],
    )

    #q_matrix_permute, k_matrix_permute, v_matrix_permute, q_matrix_bias_permute, k_matrix_bias_permute, v_matrix_bias_permute = weights_affected_rows
    #qvk_matrix_permute = torch.cat((q_matrix_permute, k_matrix_permute, v_matrix_permute), dim=0)
    #qvk_matrix_bias_permute = torch.cat((q_matrix_bias_permute, k_matrix_bias_permute, v_matrix_bias_permute), dim=0)

    q_matrix_permute, k_matrix_permute, q_matrix_bias_permute, k_matrix_bias_permute = weights_affected_rows
    qvk_matrix_permute = torch.cat((q_matrix_permute, k_matrix_permute, qvk_matrix_permute[2 * qvk_matrix_permute.shape[0] // 3:]), dim=0)
    qvk_matrix_bias_permute = torch.cat((q_matrix_bias_permute, k_matrix_bias_permute, qvk_matrix_bias_permute[2 * qvk_matrix_bias_permute.shape[0] // 3:]), dim=0)

    weights_permute[qvk_matrix_id] = qvk_matrix_permute
    weights_permute[qvk_matrix_bias_id] = qvk_matrix_bias_permute
    #weights_permute[affected_next_layer_id] = weights_affected_columns[0]

    model_permute.load_state_dict(weights_permute)


def permute_value_layer(model_true, model_permute, layer):
    weights_true = model_true.state_dict()
    weights_permute = model_permute.state_dict()

    v_matrix_id = 'transformer_encoder.layers.' + str(layer) + '.self_attn.in_proj_weight'
    v_matrix_true = weights_true[v_matrix_id]
    v_matrix_permute = weights_permute[v_matrix_id]
    v_matrix_true = v_matrix_true[2 * v_matrix_true.shape[0] // 3:]
    v_matrix_permute = v_matrix_permute[2 * v_matrix_permute.shape[0] // 3:]

    v_matrix_bias_id = 'transformer_encoder.layers.' + str(layer) + '.self_attn.in_proj_bias'
    v_matrix_bias_true = weights_true[v_matrix_bias_id]
    v_matrix_bias_permute = weights_permute[v_matrix_bias_id]
    v_matrix_bias_true = v_matrix_bias_true[2 * v_matrix_bias_true.shape[0] // 3:]
    v_matrix_bias_permute = v_matrix_bias_permute[2 * v_matrix_bias_permute.shape[0] // 3:]

    affected_next_layer_id = 'transformer_encoder.layers.' + str(layer) + '.self_attn.out_proj.weight'
    affected_next_layer_permute = model_permute.state_dict()[affected_next_layer_id]

    weights_affected_rows, weights_affected_columns, _ = permute_layer(
        [
            (v_matrix_true, v_matrix_permute),
            (v_matrix_bias_true, v_matrix_bias_permute)
        ],
        [
            v_matrix_permute,
            v_matrix_bias_permute,
        ],
        [
            affected_next_layer_permute,
        ],
        []
    )

    v_matrix_permute, v_matrix_bias_permute = weights_affected_rows
    v_matrix_permute = torch.cat((model_permute.state_dict()[v_matrix_id][:2 * model_permute.state_dict()[v_matrix_id].shape[0] // 3], v_matrix_permute), dim=0)
    v_matrix_bias_permute = torch.cat((model_permute.state_dict()[v_matrix_bias_id][:2 * model_permute.state_dict()[v_matrix_bias_id].shape[0] // 3], v_matrix_bias_permute), dim=0)

    weights_permute[v_matrix_id] = v_matrix_permute
    weights_permute[v_matrix_bias_id] = v_matrix_bias_permute
    weights_permute[affected_next_layer_id] = weights_affected_columns[0]

    model_permute.load_state_dict(weights_permute)


def permute_layer(weights_pairs_determined, weights_affected_rows, weights_affected_columns, weights_affected_third):
    # determining permutation
    permutation = get_best_permutation(weights_pairs_determined)
    print('permuted nodes ',
          np.count_nonzero(permutation - np.arange(permutation.shape[0])))

    # weights that need to be permuted
    for i in range(len(weights_affected_rows)):
        weights_affected_rows[i] = permute_rows(permutation, weights_affected_rows[i])
    for i in range(len(weights_affected_columns)):
        weights_affected_columns[i] = permute_columns(permutation, weights_affected_columns[i])
    for i in range(len(weights_affected_third)):
        weights_affected_third[i] = permute_third_dimension(permutation, weights_affected_third[i])

    return weights_affected_rows, weights_affected_columns, weights_affected_third


def permute_rows(permutation, weights):
    return weights[permutation]


def permute_columns(permutation, weights):
    return weights[:, permutation]


def permute_third_dimension(permutation, weights):
    return weights[:, :, permutation]


# Calculates the cost (difference in weights) for each node individually being mapped to another node.
# All of these costs are put into a matrix.
# Then uses linear sum assignment solver to find the best permutation
def get_best_permutation(weights_pairs):
    cost_matrix = get_cost_matrix(weights_pairs)
    _, col_idx = optimize.linear_sum_assignment(cost_matrix)
    return col_idx


def get_cost_matrix(weights_pairs):
    n = weights_pairs[0][0].shape[0]
    cost_matrix = np.zeros((n, n))
    for true_node in range(n):
        for merge_node in range(n):
            weights_pairs_node = [(weights_pair[0][true_node], weights_pair[1][merge_node]) for weights_pair in weights_pairs]
            cost_matrix[true_node, merge_node] = cost(weights_pairs_node)
    return cost_matrix


def cost(weights_pairs):
    cost_sum = 0
    for weights_true, weights_merge in weights_pairs:
        if weights_true is not None:
            cost_sum += torch.sum(torch.abs(weights_true - weights_merge))
    return cost_sum

