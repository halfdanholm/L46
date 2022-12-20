import train
import torch


def main():
    emsize = 64  # embedding dimension
    d_hid = 64  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 8  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    ntokens = 28782  # size of vocabulary

    model = train.get_model(ntokens, emsize, d_hid, nlayers, nhead, dropout)
    """model_2 = torch.load('checkpoints/model_original.pt')
    model_2 = torch.load('checkpoints/model_original.pt')"""
    train.train(model, 1)


if __name__ == '__main__':
    main()
