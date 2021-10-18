import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import data


def get_schedule(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    """From Huggingface"""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class PositionalEncoding(nn.Module):
    """From PyTorch"""

    def __init__(self, d_model, dropout=0.1, max_len=4096):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class DefmodModel(nn.Module):
    """A transformer architecture for Definition Modeling."""

    def __init__(
        self, vocab, d_model=256, n_head=4, n_layers=4, dropout=0.3, maxlen=256
    ):
        super(DefmodModel, self).__init__()
        self.d_model = d_model
        self.padding_idx = vocab[data.PAD]
        self.eos_idx = vocab[data.EOS]
        self.maxlen = maxlen

        self.embedding = nn.Embedding(len(vocab), d_model, padding_idx=self.padding_idx)
        self.positional_encoding = PositionalEncoding(
            d_model, dropout=dropout, max_len=maxlen
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dropout=dropout, dim_feedforward=d_model * 2
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.v_proj = nn.Linear(d_model, len(vocab))
        # initializing weights
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            else:  # gain parameters of the layer norm
                nn.init.ones_(param)

    def generate_square_subsequent_mask(self, sz):
        "from Pytorch"
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, vector, input_sequence=None):
        device = next(self.parameters()).device
        embs = self.embedding(input_sequence)
        seq = torch.cat([vector.unsqueeze(0), embs], dim=0)
        src = self.positional_encoding(seq)
        src_mask = self.generate_square_subsequent_mask(src.size(0)).to(device)
        src_key_padding_mask = torch.cat(
            [
                torch.tensor([[False] * input_sequence.size(1)]).to(device),
                (input_sequence == self.padding_idx),
            ],
            dim=0,
        ).t()
        transformer_output = self.transformer_encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        v_dist = self.v_proj(transformer_output)
        return v_dist

    @staticmethod
    def load(file):
        return torch.load(file)

    def save(self, file):
        file.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self, file)

    @torch.no_grad()
    def pred(self, vector, decode_fn=None, beam_size=64, verbose=False):
        # which device we should cast our variables to
        device = next(self.parameters()).device

        # how many examples are batched together
        batch_size = vector.size(0)

        # Tensors will have this shape:
        # [Sequence, Batch, Beam, Continuation, *]

        # accumulation variable, keeping track of the best beams for each batched example
        generated_symbols = torch.zeros(0, batch_size, beam_size, dtype=torch.long).to(device)

        # which beams hold a completed sequence
        current_beam_size = 1
        has_stopped = torch.tensor([False] * (batch_size * current_beam_size)).to(device)

        # the input to kick-start the generation is the embedding, we start with the same input for each beam
        vector_src = vector.unsqueeze(1).expand(batch_size, current_beam_size, -1).reshape(1,  batch_size * current_beam_size, -1)
        src = vector_src
        src_key_padding_mask = torch.tensor([[False] * (batch_size * current_beam_size)]).to(device)

        # variables needed to compute the score of each beam (geometric mean of probability of emission)
        logprobs = torch.zeros(batch_size, current_beam_size, dtype=torch.double).to(device)
        lengths = torch.zeros(batch_size * current_beam_size, dtype=torch.int).to(device)
        # generate tokens step by step
        for step_idx in range(self.maxlen):

            # generation mask
            src_mask = self.generate_square_subsequent_mask(src.size(0)).to(device)
            # positional encoding
            src_pe = self.positional_encoding(src)
            # transformer output
            transformer_output = self.transformer_encoder(
                src_pe, mask=src_mask, src_key_padding_mask=src_key_padding_mask.t()
            )[-1]
            # distribution over the full vocabulary
            v_dist = self.v_proj(transformer_output)
            # don't generate padding tokens
            v_dist[...,self.padding_idx] = -float("inf")
            v_dist = F.log_softmax(v_dist, dim=-1)

            # for each beam, select the best candidate continuations
            new_logprobs, new_symbols = v_dist.topk(beam_size, dim=-1)
            # patch the output scores to zero-out items that have already stopped
            new_logprobs = new_logprobs.masked_fill(has_stopped.unsqueeze(-1), 0.0)
            # if the beam hasn't stopped, then it needs to produce at least an EOS
            # so we can just add one to beams that have not stopped to account for the current token
            lengths += (~has_stopped).int()

            # compute scores for each continuation
            ## recreate the score of the previous full sequence for all possible continuations
            logprobs_ = logprobs.view(batch_size * current_beam_size, 1).expand(batch_size * current_beam_size, beam_size)
            ## add the cost of each continuation
            logprobs_ = logprobs_ + new_logprobs
            ## average over the full sequence, ignoring padding items
            avg_logprobs = logprobs_ #/ lengths.unsqueeze(-1)
            ## select the `beam_size` best continuations overall, their matching scores will be `avg_logprobs`
            avg_logprobs, selected_beams = avg_logprobs.view(batch_size, current_beam_size * beam_size).topk(beam_size, dim=-1)
            ## select back the base score for the selected continuations
            logprobs = logprobs_.view(batch_size, current_beam_size * beam_size).gather(-1, selected_beams).view(batch_size, beam_size)

            # add symbols of best continuations
            ## recreate the full previous sequence for all possible continuations
            generated_symbols_ = generated_symbols.view(-1, batch_size * current_beam_size, 1).expand(-1, batch_size * current_beam_size, beam_size)
            ## stack on the new symbols
            generated_symbols_ = torch.cat([generated_symbols_, new_symbols.unsqueeze(0)], dim=0)
            ## grab only the `beam_size` best continuations out of all possible continuations
            generated_symbols_ = generated_symbols_.view(-1, batch_size, current_beam_size * beam_size)
            generated_symbols = generated_symbols_.gather(-1, selected_beams.unsqueeze(0).expand(step_idx + 1, batch_size,  beam_size)).view(step_idx + 1, batch_size, beam_size)

            # recompute which beams have stopped, and what their lengths are
            ## reconstruct the lengths of all candidate continuations
            lengths = lengths.view(batch_size, current_beam_size, 1).expand(batch_size, current_beam_size, beam_size)
            ## retrieve the lengths of the selected beam continuations
            lengths = lengths.reshape(batch_size, current_beam_size * beam_size).gather(-1, selected_beams).view(-1)
            ## reconstruct the halting state of all candidate continuations
            has_stopped = has_stopped.view(batch_size, current_beam_size, 1).expand(batch_size, current_beam_size, beam_size)
            ## retrieve the halting states of selected beam continuations
            has_stopped = has_stopped.reshape(batch_size, current_beam_size * beam_size).gather(-1, selected_beams).view(-1)

            # flag which beams have terminated at the current step (i.e., whether they just produced an EOS)
            generated_symbols = generated_symbols.view(-1, batch_size * beam_size)
            generated_symbols[-1] = generated_symbols[-1].masked_fill(has_stopped, self.padding_idx)
            has_stopped = has_stopped | (generated_symbols.view(-1, batch_size * beam_size)[-1] == self.eos_idx).view(batch_size * beam_size)

            # recompute padding mask on the basis of which continuations were selected
            src_key_padding_mask = src_key_padding_mask.view(-1, batch_size, current_beam_size, 1).expand(-1, batch_size, current_beam_size, beam_size)
            src_key_padding_mask = src_key_padding_mask.reshape(-1, batch_size, current_beam_size * beam_size)
            src_key_padding_mask = src_key_padding_mask.gather(-1, selected_beams.unsqueeze(0).expand(step_idx + 1, batch_size,  beam_size)).view(step_idx + 1, batch_size * beam_size)
            src_key_padding_mask = torch.cat([src_key_padding_mask, has_stopped.unsqueeze(0)], dim=0)

            # produce input for the next timestep
            src = torch.cat([vector_src.expand(1, beam_size, -1), self.embedding(generated_symbols)], dim=0)
            # reshape to the familiar format
            generated_symbols = generated_symbols.view(-1, batch_size, beam_size)

            # if all beams have stopped, so do we
            if has_stopped.all():
                break
            # we update the number of sustained beam at the first iteration, since we know have `beam_size` candidates.
            current_beam_size = beam_size

        # select the most likely sequence for each batched item
        max_scores, selected_beams = (logprobs / lengths.view(batch_size, beam_size)).topk(1, dim=1)
        output_sequence = generated_symbols.gather(1, selected_beams.unsqueeze(0).expand(step_idx + 1, batch_size, 1))
        if verbose: print(decode_fn(output_sequence.squeeze(-1)))
        return output_sequence.squeeze(-1)


class RevdictModel(nn.Module):
    """A transformer architecture for Definition Modeling."""

    def __init__(
        self, vocab, d_model=256, n_head=4, n_layers=4, dropout=0.3, maxlen=512
    ):
        super(RevdictModel, self).__init__()
        self.d_model = d_model
        self.padding_idx = vocab[data.PAD]
        self.eos_idx = vocab[data.EOS]
        self.maxlen = maxlen

        self.embedding = nn.Embedding(len(vocab), d_model, padding_idx=self.padding_idx)
        self.positional_encoding = PositionalEncoding(
            d_model, dropout=dropout, max_len=maxlen
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dropout=dropout, dim_feedforward=d_model * 2
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.dropout = nn.Dropout(p=dropout)
        self.e_proj = nn.Linear(d_model, d_model)
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            else:  # gain parameters of the layer norm
                nn.init.ones_(param)

    def forward(self, gloss_tensor):
        src_key_padding_mask = gloss_tensor == self.padding_idx
        embs = self.embedding(gloss_tensor)
        src = self.positional_encoding(embs)
        transformer_output = self.dropout(
            self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask.t())
        )
        summed_embs = transformer_output.masked_fill(
            src_key_padding_mask.unsqueeze(-1), 0
        ).sum(dim=0)
        return self.e_proj(F.relu(summed_embs))

    @staticmethod
    def load(file):
        return torch.load(file)

    def save(self, file):
        torch.save(self, file)


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction="mean"):
    return (
        loss.mean()
        if reduction == "mean"
        else loss.sum()
        if reduction == "sum"
        else loss
    )


#  Implementation of Label smoothing with CrossEntropy and ignore_index
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction="mean", ignore_index=-100):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index
        )
        return linear_combination(loss / n, nll, self.epsilon)
