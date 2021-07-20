import collections
import datetime
import itertools
import json
import math
import os
os.environ['MOVERSCORE_MODEL'] = "distilbert-base-multilingual-cased"
import moverscore_v2 as mv_sc
import torch
import torch.nn as nn
import tqdm
import numpy as np

def mover_sentence_score(hypothesis, references, trace=0):
    """From the MoverScore github"""
    idf_dict_hyp = collections.defaultdict(lambda: 1.)
    idf_dict_ref = collections.defaultdict(lambda: 1.)
    hypothesis = [hypothesis] * len(references)
    sentence_score = 0
    scores = mv_sc.word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    sentence_score = np.mean(scores)
    if trace > 0:
        print(hypothesis, references, sentence_score)
    return sentence_score

def mover_corpus_score(sys_stream, ref_streams, trace=0):
    """From the MoverScore github"""
    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]
    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]
    fhs = [sys_stream] + ref_streams
    corpus_score = 0
    for lines in itertools.zip_longest(*fhs):
        if None in lines:
            raise EOFError("Source and reference streams have different lengths!")
        hypo, *refs = lines
        corpus_score += mover_sentence_score(hypo, refs, trace=0)
    corpus_score /= len(sys_stream)
    return corpus_score

def average(nums):
    nums = list(nums)
    return sum(nums) / len(nums)

def display(*msg):
    """Format message"""
    print(datetime.datetime.now(), *msg)

BOS_token, EOS_token, PAD_token, UNK_token = "<s>", "</s>", "<pad/>", "<unk/>"

def read_dataset(filename, vocab=collections.defaultdict(itertools.count().__next__), device=torch.device("cpu"), special_symbols=(BOS_token, EOS_token, PAD_token)):
    """Load a dataset; convert embeddings into tensors, convert gloss into id sequences"""

    # 1. Open the JSON
    with open(filename, "r") as istr:
        data = json.load(istr)

    # 2. Describe contents
    vec_archs = set(data[0].keys()) - {"id", "gloss", "word", "pos"}
    display(f"file {filename}: found these architectures:", *sorted(vec_archs))

    # 3. Ensure special tokens exist in vocab
    for symb in special_symbols:
        _ = vocab[symb]
    UNK = vocab[UNK_token]

    # 4. convert to tensors as needed
    for example in tqdm.tqdm(data, leave=False, desc="Embs."):
        example["tensors"] = {}
        for arch in vec_archs:
            example["tensors"][arch] = torch.tensor(example[arch], device=device)
    if "gloss" in data[0].keys():
        display(f"file {filename}: found a gloss")
        prefix = [vocab[BOS_token]] if BOS_token in special_symbols else []
        suffix = [vocab[BOS_token]] if EOS_token in special_symbols else []
        for example in tqdm.tqdm(data, leave=False, desc="Glosses"):
            try:
                gloss = prefix + [vocab[w] for w in example["gloss"].split()] + suffix
                example["tensors"]["gloss"] = torch.tensor(gloss, device=device)
            except KeyError:
                # vocab has been frozen and we stumbled upon an OOV
                gloss = prefix + [vocab.get(w, UNK) for w in example["gloss"].split()] + suffix
                example["tensors"]["gloss"] = torch.tensor(gloss, device=device)

    # 5. freeze vocab and return encoded data
    return data, dict(vocab)


# useful torch modules

class PositionalEncoding(nn.Module):
    "From PyTorch (https://github.com/pytorch/examples/blob/master/word_language_model/model.py#L65)"

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Assuming x is a sequence of ids"""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SumLayer(nn.Module):
    """A simple layer summing the input"""
    def __init__(self, dim=0):
        # we're supposing no batching, no padding, no nothing.
        super(SumLayer, self).__init__()
        self.dim = dim

    def forward(self, sequence):
        """sum along the dimension"""
        return sequence.sum(dim=self.dim)

class SubsequentMasked(nn.Module):
    """A simple wrapper for masking transformer hidden states"""
    def __init__(self, module):
        super(SubsequentMasked, self).__init__()
        self.module = module

    def generate_square_subsequent_mask(self, sz):
        "from Pytorch"
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        # we're supposing no batching, no padding, no nothing.
        mask = self.generate_square_subsequent_mask(x.size(0))
        return self.module(x, mask=mask)
