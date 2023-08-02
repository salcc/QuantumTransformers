import os
from collections import Counter

import torch
import torch.utils.data
import torchtext

from quantum_transformers.datasets import datasets_to_dataloaders


def get_imdb_dataloaders(root: str = '~/data', **dataloader_kwargs) \
        -> tuple[tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader], torchtext.vocab.Vocab]:
    """
    Returns dataloaders for the IMDB sentiment analysis dataset (natural language processing, binary classification),
    along with the vocabulary object.
    """
    root = os.path.expanduser(root)
    train_dataset = torchtext.datasets.IMDB(root, split='train')
    valid_dataset = torchtext.datasets.IMDB(root, split='test')

    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    counter: Counter[str] = Counter()
    for _, line in train_dataset:
        counter.update(tokenizer(line))
    unk_token, bos_token, eos_token, pad_token = '<UNK>', '<BOS>', '<EOS>', '<PAD>'
    vocab = torchtext.vocab.vocab(counter, min_freq=10, specials=[unk_token, bos_token, eos_token, pad_token])
    vocab.set_default_index(vocab[unk_token])

    def collate_batch(batch):
        label_list, text_list = [], []
        for label, text in batch:
            label_list.append(label - 1)  # 1/2 -> 0/1
            text_list.append(torch.tensor([vocab['<BOS>']] + [vocab[token] for token in tokenizer(text)] + [vocab['<EOS>']]))
        return torch.nn.utils.rnn.pad_sequence(text_list, padding_value=vocab['<PAD>'], batch_first=True), torch.tensor(label_list)

    return (datasets_to_dataloaders(list(train_dataset), list(valid_dataset), collate_fn=collate_batch, **dataloader_kwargs), vocab)  # type: ignore
