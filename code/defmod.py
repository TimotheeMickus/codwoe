import argparse
import itertools
import json
import logging
import pathlib
import sys

logger = logging.getLogger(pathlib.Path(__file__).name)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
logger.addHandler(handler)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import tqdm

import data
import models


def get_parser(parser=argparse.ArgumentParser(description="run a definition modeling baseline")):
    parser.add_argument("--do_train", action="store_true", help="whether to train a model from scratch")
    parser.add_argument("--do_pred", action="store_true", help="whether to produce predictions")
    parser.add_argument("--train_file", type=pathlib.Path, help="path to the train file")
    parser.add_argument("--dev_file", type=pathlib.Path, help="path to the dev file")
    parser.add_argument("--test_file", type=pathlib.Path, help="path to the test file")
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"), help="path to the train file")
    parser.add_argument("--source_arch", type=str, default="sgns", choices=("sgns", "char", "electra"), help="embedding architecture to use as source")
    parser.add_argument("--summary_logdir", type=pathlib.Path, default=pathlib.Path("logs") / "defmod-baseline", help="write logs for future analysis")
    parser.add_argument("--save_dir", type=pathlib.Path, default=pathlib.Path("models") / "defmod-baseline", help="where to save model & vocab")
    parser.add_argument("--pred_file", type=pathlib.Path, default=pathlib.Path("defmod-baseline-preds.json"), help="where to save predictions")
    return parser


def train(args):
    assert args.train_file is not None, "Missing dataset for training"
    # 1. get data, vocabulary, summary writer
    logger.debug("Preloading training data")
    ## make datasets
    train_dataset = data.JSONDataset(args.train_file)
    if args.dev_file:
        dev_dataset = data.JSONDataset(args.dev_file, vocab=train_dataset.vocab)
    ## assert they correspond to the task
    assert train_dataset.has_gloss, "Training dataset contains no gloss."
    if args.source_arch == "electra":
        assert train_dataset.has_electra, \
            "Training datatset contains no vector."
    else:
        assert train_dataset.has_vecs, "Training datatset contains no vector."
    if args.dev_file:
        assert dev_dataset.has_gloss, "Development dataset contains no gloss."
        if args.source_arch == "electra":
            assert dev_dataset.has_electra, \
                "Development dataset contains no vector."
        else:
            assert dev_dataset.has_vecs, \
                "Development dataset contains no vector."
    ## make dataloader
    train_dataloader = data.get_dataloader(train_dataset)
    dev_dataloader = data.get_dataloader(dev_dataset, shuffle=False)
    ## make summary writer
    summary_writer = SummaryWriter(args.summary_logdir)
    train_step = itertools.count() # to keep track of the training steps for logging

    # 2. construct model
    logger.debug("Setting up training environment")

    model = models.DefmodModel(dev_dataset.vocab).to(args.device)
    model.train()

    # 3. declare optimizer & criterion
    ## Hyperparams
    EPOCHS, LEARNING_RATE, BETA1, BETA2, WEIGHT_DECAY = 50, 1.e-4, .9, .999, 1.e-5
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(ignore_index=model.padding_idx)

    vec_tensor_key = f"{args.source_arch}_tensor"

    # 4. train model
    for epoch in tqdm.trange(EPOCHS, desc="Epoch", disable=None):
        ## train loop
        pbar = tqdm.tqdm(desc=f"Train {epoch}", total=len(train_dataset), disable=None)
        for batch in train_dataloader:
            optimizer.zero_grad()
            vec = batch[vec_tensor_key].to(args.device)
            gls = batch["gloss_tensor"].to(args.device)
            pred = model(vec, gls[:-1])
            loss = criterion(pred.view(-1, pred.size(-1)), gls.view(-1))
            loss.backward()
            # keep track of the train loss for this step
            tokens = (gls != model.padding_idx)
            acc = (((pred.argmax(-1) == gls) & tokens).float().sum() / tokens.sum()).item()
            step = next(train_step)
            summary_writer.add_scalar("train/loss", loss.item(), step)
            summary_writer.add_scalar("train/acc", acc, step)
            optimizer.step()
            pbar.update(vec.size(0))
        ## eval loop
        if args.dev_file:
            model.eval()
            with torch.no_grad():
                sum_dev_loss = 0.0
                sum_acc = 0
                ntoks = 0
                pbar = tqdm.tqdm(desc=f"Eval {epoch}", total=len(dev_dataset), disable=None)
                for batch in dev_dataloader:
                    vec = batch[vec_tensor_key].to(args.device)
                    gls = batch["gloss_tensor"].to(args.device)
                    pred = model(vec, gls[:-1])
                    sum_dev_loss += F.cross_entropy(
                        pred.view(-1, pred.size(-1)),
                        gls.view(-1),
                        reduction="sum",
                        ignore_index=model.padding_idx
                    ).item()
                    tokens = (gls != model.padding_idx)
                    ntoks += tokens.sum().item()
                    sum_acc += ((pred.argmax(-1) == gls) & tokens).sum().item()
                    pbar.update(vec.size(0))

                # keep track of the average loss & acc on dev set for this epoch
                summary_writer.add_scalar("dev/loss", sum_dev_loss/ntoks, epoch)
                summary_writer.add_scalar("dev/acc", sum_acc/ntoks, epoch)
            model.train()

    # 5. save result
    model.save(args.save_dir / "model.pt")
    train_dataset.save(args.save_dir / "train_dataset.pt")
    dev_dataset.save(args.save_dir / "dev_dataset.pt")


def pred(args):
    assert args.test_file is not None, "Missing dataset for test"
    # 1. retrieve vocab, dataset, model
    model = models.DefmodModel.load(args.save_dir / "model.pt")
    train_vocab = data.JSONDataset.load(args.save_dir / "train_dataset.pt").vocab
    test_dataset = data.JSONDataset(args.test_file, vocab=train_vocab, freeze_vocab=True, maxlen=model.maxlen)
    test_dataloader = data.get_dataloader(test_dataset, shuffle=False)
    model.eval()
    vec_tensor_key = f"{args.source_arch}_tensor"
    if args.source_arch == "electra":
        assert test_dataset.has_electra, "File is not usable for the task"
    else:
        assert test_dataset.has_vecs, "File is not usable for the task"
    # 2. make predictions
    predictions = []
    with torch.no_grad():
        pbar = tqdm.tqdm(desc="Pred.", total=len(test_dataset), disable=None)
        for batch in test_dataloader:
            sequence = model.pred(batch[vec_tensor_key].to(args.device))
            for id, gloss in zip(batch["id"], test_dataset.decode(sequence)):
                predictions.append({"id": id, "gloss": gloss})
            pbar.update(batch[vec_tensor_key].size(0))
    # 3. dump predictions
    with open(args.pred_file, "w") as ostr:
        json.dump(predictions, ostr)


def main(args):
    if args.do_train:
        logger.debug("Performing defmod training")
        train(args)
    if args.do_pred:
        logger.debug("Performing defmod prediction")
        pred(args)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
