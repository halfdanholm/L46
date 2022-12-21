######################################################################
# Load and batch data
# -------------------
#
import torchtext

######################################################################
# This tutorial uses ``torchtext`` to generate Wikitext-2 dataset.
# To access torchtext datasets, please install torchdata following instructions at https://github.com/pytorch/data.
# %%
#  .. code-block:: bash
#
#      %%bash
#      pip install torchdata
#
# The vocab object is built based on the train dataset and is used to numericalize
# tokens into tensors. Wikitext-2 represents rare tokens as `<unk>`.
#
# Given a 1-D vector of sequential data, ``batchify()`` arranges the data
# into ``batch_size`` columns. If the data does not divide evenly into
# ``batch_size`` columns, then the data is trimmed to fit. For instance, with
# the alphabet as the data (total length of 26) and ``batch_size=4``, we would
# divide the alphabet into 4 sequences of length 6:
#
# .. math::
#   \begin{bmatrix}
#   \text{A} & \text{B} & \text{C} & \ldots & \text{X} & \text{Y} & \text{Z}
#   \end{bmatrix}
#   \Rightarrow
#   \begin{bmatrix}
#   \begin{bmatrix}\text{A} \\ \text{B} \\ \text{C} \\ \text{D} \\ \text{E} \\ \text{F}\end{bmatrix} &
#   \begin{bmatrix}\text{G} \\ \text{H} \\ \text{I} \\ \text{J} \\ \text{K} \\ \text{L}\end{bmatrix} &
#   \begin{bmatrix}\text{M} \\ \text{N} \\ \text{O} \\ \text{P} \\ \text{Q} \\ \text{R}\end{bmatrix} &
#   \begin{bmatrix}\text{S} \\ \text{T} \\ \text{U} \\ \text{V} \\ \text{W} \\ \text{X}\end{bmatrix}
#   \end{bmatrix}
#
# Batching enables more parallelizable processing. However, batching means that
# the model treats each column independently; for example, the dependence of
# ``G`` and ``F`` can not be learned in the example above.
#

from transformer import *

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import copy
import time


def data_process(raw_text_iter: dataset.IterableDataset, vocab, tokenizer) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: Tensor, bsz: int, device) -> Tensor:
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


######################################################################
# Functions to generate input and target sequence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# ``get_batch()`` generates a pair of input-target sequences for
# the transformer model. It subdivides the source data into chunks of
# length ``bptt``. For the language modeling task, the model needs the
# following words as ``Target``. For example, with a ``bptt`` value of 2,
# we’d get the following two Variables for ``i`` = 0:
#
# .. image:: ../_static/img/transformer_input_target.png
#
# It should be noted that the chunks are along dimension 0, consistent
# with the ``S`` dimension in the Transformer model. The batch dimension
# ``N`` is along dimension 1.
#


def get_batch(source: Tensor, i: int, bptt: int) -> Tuple[Tensor, Tensor]:
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


######################################################################
# Initiate an instance
# --------------------
#


######################################################################
# The model hyperparameters are defined below. The vocab size is
# equal to the length of the vocab object.
#

"""
emsize = 64  # embedding dimension
d_hid = 64  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
"""


def get_model(ntokens: int, emsize: int, nhead: int, d_hid: int, nlayers: int, dropout: float) -> TransformerModel:
    return TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout)


def get_weights(model):
    weights_1 = model.transformer_encoder.layers[1].weight
    biases_1 = model.transformer_encoder.layers[1].bias
    weights_0 = model.transformer_encoder.layers[0].weight
    biases_0 = model.transformer_encoder.layers[0].bias
    return weights_1, biases_1, weights_0, biases_0


# How they get the weights in FedMA: statedict = net.state_dict()


######################################################################
# Run the model
# -------------
#


######################################################################
# We use `CrossEntropyLoss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>`__
# with the `SGD <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`__
# (stochastic gradient descent) optimizer. The learning rate is initially set to
# 5.0 and follows a `StepLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html>`__
# schedule. During training, we use `nn.utils.clip_grad_norm\_ <https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>`__
# to prevent gradients from exploding.
#


def train_epoch(model: nn.Module, ntokens: int, epoch: int, criterion, optimizer, scheduler, train_data, bptt,
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


def evaluate(model: nn.Module, eval_data: Tensor, device, ntokens: int = 28782, criterion=nn.CrossEntropyLoss(),
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


"""# Generates text from a model. Generated with Codex
def generate_text(model: nn.Module, prompt: str, num_words: int, temperature: float = 1.0) -> str:
    model.eval()  # turn on evaluation mode
    with torch.no_grad():
        # Generate a single word
        def generate_word(prev_word: int, prev_state: Tuple[Tensor, Tensor]) -> Tuple[int, Tuple[Tensor, Tensor]]:
            # Get the previous word embedding
            prev_word_embed = model.src_embed(prev_word)
            # Get the previous state
            prev_state = model.transformer.decoder.layers[0].self_attn.prev_state
            # Get the next state
            next_state = model.transformer.decoder.layers[0].self_attn(prev_word_embed, prev_state)
            # Get the output
            output = model.transformer.decoder.layers[0].self_attn.output
            # Get the logits
            logits = model.transformer.decoder.layers[0].self_attn.logits
            # Get the probabilities
            probs = F.softmax(logits / temperature, dim=-1)
            # Sample the next word
            next_word = torch.multinomial(probs, num_samples=1).item()
            return next_word, next_state"""


######################################################################
# Loop over epochs. Save the model if the validation loss is the best
# we've seen so far. Adjust the learning rate after each epoch.

def get_dataset_split(device):
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
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


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')


def train(model: nn.Module, train_data, device, name: str = "1", epochs: int = 1, ntokens: int = 28782) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
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
