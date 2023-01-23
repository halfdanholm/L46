import copy
import merge
import transformer
import torch
import sys
import argparse
import data


def main():
    emsize = 4  # embedding dimension
    d_hid = 6  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 1  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 1  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    ntokens = 28782  # size of vocabulary

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoints_folder", type=str, default="skip", help="Checkpoint Folder")
    parser.add_argument("--data_type", type=str, default="hetero")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--diff_init", action="store_true")
    parser.add_argument("--lr", type=int, default=5.0)
    args = parser.parse_args()

    device = transformer.get_device()
    data_1_orig, data_2_orig, val_data_orig, _ = data.get_original_dataset_split(device)
    data_1, data_2, val_data, _ = data.get_dataset_split(device, type=args.data_type, batch_size=args.batch_size)
    val_data = val_data[:100]

    if args.checkpoints_folder != 'skip':
        print('Loading checkpoints...')
        model_1_trained = torch.load(f'{sys.argv[2]}/model_1.pt', map_location=device)
        model_2_trained = torch.load(f'{sys.argv[2]}/model_2.pt', map_location=device)
        model_1_trained.to(device)
        model_2_trained.to(device)
    else:
        model_1 = transformer.get_model(ntokens, emsize, nhead, d_hid, nlayers, dropout)
        if args.diff_init:
            model_2 = transformer.get_model(ntokens, emsize, nhead, d_hid, nlayers, dropout)
        else:
            model_2 = copy.deepcopy(model_1)
        print('Training model 1...')
        model_1_trained = transformer.train(model_1, data_1, device, name='1', epochs=args.epochs, lr=args.lr)
        print('Training model 2...')
        model_2_trained = transformer.train(model_2, data_2, device, name='2', epochs=args.epochs, lr=args.lr)

    print(model_1_trained)
    print('Got models')

    model_permuted = merge.permute_model(device, model_1_trained, model_2_trained)
    model_permuted.to(device)

    model_merged = merge.average_model(model_1_trained, model_permuted)
    model_merged.to(device)

    model_av = merge.average_model(model_1_trained, model_2_trained)
    model_av.to(device)

    loss_merged = transformer.evaluate(model_merged, val_data, device)
    print(f'Loss merged: {loss_merged}')
    loss_av = transformer.evaluate(model_av, val_data, device)
    print(f'Loss average: {loss_av}')
    loss_1 = transformer.evaluate(model_1_trained, val_data, device)
    print(f'Loss 1: {loss_1}')
    loss_2 = transformer.evaluate(model_2_trained, val_data, device)
    print(f'Loss 2: {loss_2}')
    loss_permuted = transformer.evaluate(model_permuted, val_data, device)
    print(f'Loss permuted: {loss_permuted}')


if __name__ == '__main__':
    main()
