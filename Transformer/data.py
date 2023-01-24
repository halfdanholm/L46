import itertools
import math
import torch
import time
import torchtext
import typing
import numpy as np


def get_dataset_split(device, type='wiki', hetero_split=False, batch_size=20):
    train_iter_wiki = torchtext.datasets.WikiText2(split='train')
    train_iter_penn = torchtext.datasets.WikiText2(split='train')
    train_iter = itertools.chain(train_iter_wiki, train_iter_penn)
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    train_iter_penn, val_iter_penn, test_iter_penn = torchtext.datasets.PennTreebank()
    train_data_penn = data_process(train_iter_penn, vocab, tokenizer)
    val_data_penn = data_process(val_iter_penn, vocab, tokenizer)
    test_data_penn = data_process(test_iter_penn, vocab, tokenizer)

    train_iter_wiki, val_iter_wiki, test_iter_wiki = torchtext.datasets.WikiText2()
    train_data_wiki = data_process(train_iter_wiki, vocab, tokenizer)
    val_data_wiki = data_process(val_iter_wiki, vocab, tokenizer)
    test_data_wiki = data_process(test_iter_wiki, vocab, tokenizer)
    if not type == 'wiki':
        train_data_wiki = train_data_wiki[:train_data_penn.shape[0]]
        val_data_wiki = train_data_wiki[:val_data_penn.shape[0]]
        test_data_wiki = train_data_wiki[:test_data_penn.shape[0]]

    eval_batch_size = 2
    train_data_wiki = batchify(train_data_wiki, batch_size, device)  # shape [seq_len, batch_size]
    val_data_wiki = batchify(val_data_wiki, eval_batch_size, device)
    test_data_wiki = batchify(test_data_wiki, eval_batch_size, device)

    train_data_penn = batchify(train_data_penn, batch_size, device)  # shape [seq_len, batch_size]
    val_data_penn = batchify(val_data_penn, eval_batch_size, device)
    test_data_penn = batchify(test_data_penn, eval_batch_size, device)

    if type == 'wiki':
        data_1, data_2, _ = train_data_wiki.split(train_data_wiki.size(0) // 2, dim=0)
        val_data = val_data_wiki
        test_data = test_data_wiki
    elif type == 'penn':
        data_1, data_2, _ = train_data_penn.split(train_data_penn.size(0) // 2, dim=0)
        val_data = val_data_penn
        test_data = test_data_penn
    elif type == 'hetero':
        data_1 = train_data_wiki
        data_2 = train_data_penn
        val_data = torch.cat((val_data_wiki[:210], val_data_penn[:210]), dim=1)
        test_data = torch.cat((test_data_wiki, test_data_penn), dim=1)
    else:
        raise ValueError('Invalid type')

    if hetero_split:
        train_data = torch.cat((data_1, data_2), dim=1)
        datas = train_data.split(train_data.size(0) // 10000, dim=0)
        averages = [np.array(data.cpu()).mean() for data in datas]
        # get the indices of the 50% of the data with the highest average and the 50% with the lowest average
        indices = np.argsort(averages)
        low_datas = [datas[i] for i in indices[:len(indices) // 2]]
        high_datas = [datas[i] for i in indices[len(indices) // 2:]]
        data_1 = torch.cat(low_datas, dim=0)
        data_2 = torch.cat(high_datas, dim=0)

    return data_1, data_2, val_data, test_data


def get_original_dataset_split(device):
    train_iter = torchtext.datasets.WikiText2(split='train')
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = torchtext.datasets.WikiText2()
    train_data = data_process(train_iter, vocab, tokenizer)
    val_data = data_process(val_iter, vocab, tokenizer)
    test_data = data_process(test_iter, vocab, tokenizer)

    batch_size = 20
    eval_batch_size = 10
    train_data = batchify(train_data, batch_size, device)  # shape [seq_len, batch_size]
    val_data = batchify(val_data, eval_batch_size, device)
    test_data = batchify(test_data, eval_batch_size, device)

    data_1, data_2, _ = train_data.split(train_data.size(0) // 2, dim=0)
    return data_1, data_2, val_data, test_data


def data_process(raw_text_iter: torch.utils.data.dataset.IterableDataset, vocab, tokenizer) -> torch.Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: torch.Tensor, bsz: int, device) -> torch.Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)
