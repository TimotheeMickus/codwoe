from collections import defaultdict
from itertools import count
import json
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

BOS="<seq>"
EOS="</seq>"
PAD="<pad/>"
UNK="<unk/>"

SUPPORTED_ARCHS = ("sgns", "char")

# A dataset is a container object for the actual data
class JSONDataset(Dataset):
    """Reads a CODWOE JSON dataset"""
    def __init__(self, file, vocab=None, freeze_vocab=False, maxlen=128):
        """
        Construct a torch.utils.data.Dataset compatible with torch data API and
        codwoe data.
        args: `file` the path to the dataset file
              `vocab` a dictionary mapping strings to indices
              `freeze_vocab` whether to update vocabulary, or just replace unknown items with OOV token
              `maxlen` the maximum number of tokens per gloss
        """
        if vocab is None:
            self.vocab = defaultdict(count().__next__)
        else:
            self.vocab = defaultdict(count(len(vocab)).__next__)
            self.vocab.update(vocab)
        pad, eos, bos, unk = self.vocab[PAD], self.vocab[EOS], self.vocab[BOS], self.vocab[UNK]
        if freeze_vocab:
            self.vocab = dict(vocab)
        with open(file, "r") as istr:
            self.items = json.load(istr)
        # preparse data
        for json_dict in self.items:
            # in definition modeling test datasets, gloss targets are absent
            if "gloss" in  json_dict:
                json_dict["gloss_tensor"] = torch.tensor([bos] + [
                    self.vocab[word] if not freeze_vocab else self.vocab.get(word, unk)
                    for word in json_dict["gloss"].split()
                ] + [eos])
                if maxlen:
                    json_dict["gloss_tensor"] = json_dict["gloss_tensor"][:maxlen]
            # in reverse dictionary test datasets, vector targets are absent
            for arch in SUPPORTED_ARCHS:
                if arch in json_dict:
                    json_dict[f"{arch}_tensor"] = torch.tensor(json_dict[arch])
            if "electra" in json_dict:
                json_dict["electra_tensor"] = torch.tensor(json_dict["electra"])
        self.has_gloss = "gloss" in self.items[0]
        self.has_vecs = SUPPORTED_ARCHS[0] in self.items[0]
        self.has_electra = "electra" in self.items[0]
        self.itos = sorted(self.vocab, key=lambda w: self.vocab[w])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

        # we're adding this method to simplify the code in our predictions of
        # glosses
    def decode(self, tensor):
        """Convert a sequence of indices (possibly batched) to tokens"""
        with torch.no_grad():
            if tensor.dim() == 2:
                # we have batched tensors of shape [Seq x Batch]
                decoded = []
                for tensor_ in tensor.t():
                    decoded.append(self.decode(tensor_))
                return decoded
            else:
                return " ".join([
                    self.itos[i.item()]
                    for i in tensor
                    if i != self.vocab[PAD]
                ])

    def save(self, file):
        torch.save(self, file)

    @staticmethod
    def load(file):
        return torch.load(file)

# A sampler allows you to define how to select items from your Dataset. Torch
# provides a number of default Sampler classes
class TokenSampler(Sampler):
    """Produce batches with up to `batch_size` tokens in each batch"""
    def __init__(self, dataset, batch_size=32, size_fn=len, drop_last=False,
    shuffle=True):
    """
    args: `dataset` a torch.utils.data.Dataset (iterable style)
          `batch_size` the maximum number of tokens in a batch
          `size_fn` a callable that yields the number of tokens in a dataset item
          `drop_last` if True and the data can't be divided in exactly the right number of batch, drop the last batch
          `shuffle` if True, shuffle between every iteration
    """
        self.dataset = dataset
        self.batch_size = batch_size
        self.size_fn = size_fn
        self._len = None
        self.drop_last = drop_last
        self.shuffle = True

    def __iter__(self):
        indices = range(len(self.dataset))
        if self.shuffle:
            indices = list(indices)
            random.shuffle(indices)
        i = 0
        selected = []
        numel = 0
        longest_len = 0
        for i in indices:
            if numel + self.size_fn(self.dataset[i]) > self.batch_size:
                yield selected
                selected = []
                numel = 0
            numel += self.size_fn(self.dataset[i])
            selected.append(i)
        if selected and not self.drop_last:
            yield selected

    def __len__(self):
        if self._len is None:
            self._len = sum(
                self.sizing_fn(self.dataset[i])
                for i in range(len(self.dataset))
            ) // self.batch_size
        return self._len

# DataLoaders give access to an iterator over the dataset, using a sampling
# strategy as defined through a Sampler.
def get_dataloader(dataset, batch_size=128, shuffle=True):
    """produce dataloader.
    args: `dataset` a torch.utils.data.Dataset (iterable style)
          `batch_size` the maximum number of tokens in a batch
          `shuffle` if True, shuffle between every iteration
    """
    # some constants for the closures
    has_gloss = dataset.has_gloss
    has_vecs = dataset.has_vecs
    has_electra = dataset.has_electra
    PAD_idx = dataset.vocab[PAD]

    # the collate function has to convert a list of dataset items into a batch
    def do_collate(json_dicts):
        """collates example into a dict batch; produces ands pads tensors"""
        batch = defaultdict(list)
        for jdict in json_dicts:
            for key in jdict:
                batch[key].append(jdict[key])
        if has_gloss:
            batch["gloss_tensor"] = pad_sequence(
                batch["gloss_tensor"],
                padding_value=PAD_idx,
                batch_first=False
            )
        if has_vecs:
            for arch in SUPPORTED_ARCHS:
                batch[f"{arch}_tensor"] = torch.stack(batch[f"{arch}_tensor"])
        if has_electra:
            batch["electra_tensor"] = torch.stack(batch["electra_tensor"])
        return dict(batch)
    if dataset.has_gloss:
        # we try to keep the amount of gloss tokens roughly constant across all
        # batches.
        def do_size_item(item):
            """retrieve tensor size, so as to batch items per elements"""
            return item["gloss_tensor"].numel()

        return DataLoader(
            dataset,
            collate_fn=do_collate,
            batch_sampler=TokenSampler(
                dataset,
                batch_size=batch_size,
                size_fn=do_size_item,
                shuffle=shuffle
            ),
        )
    else:
        # there's no gloss, hence no gloss tokens, so we use a default batching
        # strategy.
        return DataLoader(
            dataset,
            collate_fn=do_collate,
            batch_size=batch_size,
            shuffle=shuffle
        )
