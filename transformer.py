import itertools
import math
import torch
import time
import torchtext
import typing
import numpy as np


class TransformerModel(torch.nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = torch.nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = torch.nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


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


def get_batch(source: torch.Tensor, i: int, bptt: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target


def get_model(ntokens: int, emsize: int, nhead: int, d_hid: int, nlayers: int, dropout: float) -> TransformerModel:
    return TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout)


def get_weights(model):
    weights_1 = model.transformer_encoder.layers[1].weight
    biases_1 = model.transformer_encoder.layers[1].bias
    weights_0 = model.transformer_encoder.layers[0].weight
    biases_0 = model.transformer_encoder.layers[0].bias
    return weights_1, biases_1, weights_0, biases_0


def train_epoch(model: torch.nn.Module, ntokens: int, epoch: int, criterion, optimizer, scheduler, train_data, bptt,
                device) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate(model: torch.nn.Module, eval_data: torch.Tensor, device, ntokens: int = 28782, criterion=torch.nn.CrossEntropyLoss(),
             bptt: int = 35) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i, bptt)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


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
    train_data_wiki = data_process(train_iter_wiki, vocab, tokenizer)[:train_data_penn.shape[0]]
    val_data_wiki = data_process(val_iter_wiki, vocab, tokenizer)[:val_data_penn.shape[0]]
    test_data_wiki = data_process(test_iter_wiki, vocab, tokenizer)[:test_data_penn.shape[0]]

    eval_batch_size = 10
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
        val_data = torch.cat((val_data_wiki, val_data_penn), dim=1)
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


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')


def train(model: torch.nn.Module, train_data, device, name: str = "1", epochs: int = 1, ntokens: int = 28782) -> torch.nn.Module:
    criterion = torch.nn.CrossEntropyLoss()
    lr = 5.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    bptt = 35

    model.to(device)
    for epoch in range(1, epochs + 1):
        train_epoch(model, ntokens, epoch, criterion, optimizer, scheduler, train_data, bptt, device)
        scheduler.step()
    torch.save(model, f'checkpoints/model_{name}.pt')

    return model
