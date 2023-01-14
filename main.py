import copy
import merge
import transformer
import torch
import sys
import argparse

def main():
    emsize = 64  # embedding dimension
    d_hid = 64  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 128  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 8  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    ntokens = 28782  # size of vocabulary

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint_folder", type=str, default="skip", help="Checkpoint Folder")
    parser.add_argument("--data_type", type=str, default="hetero")
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    device = transformer.get_device()
    data_1, data_2, val_data, _ = transformer.get_dataset_split(device, type=args.data_type, batch_size=args.batch_size)
    model_1 = transformer.get_model(ntokens, emsize, d_hid, nlayers, nhead, dropout)
    model_2 = copy.deepcopy(model_1)

    if args.checkpoint_folder != 'skip':
        model_1_trained = torch.load(f'{sys.argv[2]}/model_1.pt')
        model_2_trained = torch.load(f'{sys.argv[2]}/model_2.pt')
        model_1_trained.to(device)
        model_2_trained.to(device)
    else:
        model_1_trained = transformer.train(model_1, data_1, device, name='1', epochs=args.epochs)
        model_2_trained = transformer.train(model_2, data_2, device, name='2', epochs=args.epochs)

    print('loaded models')

    model_av = merge.average_model(model_1_trained, model_2_trained)
    model_av.to(device)

    model_aav = merge.almost_average_model(model_1_trained, model_2_trained)
    model_aav.to(device)

    loss_1 = transformer.evaluate(model_1_trained, val_data, device)
    print(f'Loss 1: {loss_1}')
    loss_2 = transformer.evaluate(model_2_trained, val_data, device)
    print(f'Loss 2: {loss_2}')
    loss_av = transformer.evaluate(model_av, val_data, device)
    print(f'Loss average: {loss_av}')
    loss_aav = transformer.evaluate(model_aav, val_data, device)
    print(f'Loss almost average: {loss_aav}')


if __name__ == '__main__':
    main()
