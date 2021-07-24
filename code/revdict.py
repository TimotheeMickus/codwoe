import argparse
import itertools
import json
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import tqdm

import utils

def get_parser(parser=argparse.ArgumentParser(description="Run a reverse dictionary baseline.\nThe task consists in reconstructing an embedding from the glosses listed in the datasets")):
    parser.add_argument("--do-train", action="store_true", help="whether to train a model from scratch")
    parser.add_argument("--do-pred", action="store_true", help="whether to produce predictions")
    parser.add_argument("--train-file", type=pathlib.Path, help="path to the train file")
    parser.add_argument("--dev-file", type=pathlib.Path, help="path to the dev file")
    parser.add_argument("--test-file", type=pathlib.Path, help="path to the test file")
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"), help="path to the train file")
    parser.add_argument("--target-arch", type=str, default="sgns", choices=("sgns", "char", "electra"), help="embedding architecture to use as target")
    parser.add_argument("--summary-logdir", type=pathlib.Path, default=pathlib.Path("logs") / f"revdict-baseline", help="write logs for future analysis")
    parser.add_argument("--save-dir", type=pathlib.Path, default=pathlib.Path("models") / f"revdict-baseline", help="where to save model & vocab")
    parser.add_argument("--pred-file", type=pathlib.Path, default=pathlib.Path("revdict-baseline-preds.json"), help="where to save predictions")
    return parser


def train(args):
    assert args.train_file is not None, "Missing dataset for training"
    # 1. get data, vocabulary, summary writer
    utils.display("Preloading training data")
    ## make datasets
    train_dataset = data.JSONDataset(args.train_file)
    if args.dev_file:
        dev_dataset = data.JSONDataset(args.dev_file, vocab=train_dataset.vocab)
    ## assert they correspond to the task
    assert train_dataset.has_gloss, "Training dataset contains no gloss."
    if args.target_arch == "electra":
        assert train_dataset.has_electra, \
            "Training datatset contains no vector."
    else:
        assert train_dataset.has_vecs, "Training datatset contains no vector."
    if args.dev_file:
        assert dev_dataset.has_gloss, "Development dataset contains no gloss."
        if args.target_arch == "electra":
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
    ## Hyperparams
    model = models.RevdictModel(dev_dataset.vocab).to(args.device)
    model.train()

    # 3. declare optimizer & criterion
    ## Hyperparams
    EPOCHS, LEARNING_RATE, BETA1, BETA2, WEIGHT_DECAY = 10, 1.e-4, .9, .999, 1.e-5
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    vec_tensor_key = f"{args.target_arch}_tensor"

    # 4. train model
    for epoch in tqdm.trange(EPOCHS, desc="Epoch"):
        ## train loop
        pbar = tqdm.tqdm(desc="Train", total=len(train_dataset), leave=False)
        for batch in train_dataloader:
            optimizer.zero_grad()
            src = batch["gloss_tensor"].to(args.device)
            tgt = batch[vec_tensor_key].to(args.device)
            pred = model(src)
            loss = criterion(pred, tgt)
            loss.backward()
            # keep track of the train loss for this step
            summary_writer.add_scalar("train/loss", loss.item(), next(train_step))
            optimizer.step()
        ## eval loop
        if args.dev_file:
            model.eval()
            with torch.no_grad():
                sum_dev_loss = 0.0
                pbar = tqdm.tqdm(desc="Train", total=len(train_dataset), leave=False)
                for batch in dev_dataloader:
                    src = batch["gloss_tensors"].to(args.device)
                    tgt = batch[vec_tensor_key].unsqueeze(0)
                    pred = model(src)
                    sum_dev_loss += criterion(pred, tgt).item()
                # keep track of the average loss on dev set for this epoch
                summary_writer.add_scalar("dev/loss", sum_dev_loss / len(dev_dataset), epoch)
            model.train()

    # 5. save result
    model.save(args.save_dir / "model.pt")
    train_dataset.save(args.save_dir / "train_dataset.pt")
    dev_dataset.save(args.save_dir / "dev_dataset.pt")


def pred(args):
    assert args.test_file is not None, "Missing dataset for test"
    # 1. retrieve vocab, dataset, model
    model = models.DefmodModel.load(args.save_dir / "model.pt")
    train_vocab = data.JSONDataset.load(args.save_dir / "train_data.pt").vocab
    test_dataset = data.JSONDataset(args.test_file, vocab=vocab)
    test_dataloader = data.get_dataloader(test_dataset, shuffle=False)
    model.eval()
    vec_tensor_key = f"{args.target_arch}_tensor"
    assert test_dataset.has_gloss, "File is not usable for the task"
    # 2. make predictions
    predictions = []
    with torch.no_grad():
        pbar = tqdm.tqdm(desc="Pred.", total=len(test_dataset))
        for batch in test_dataloader:
            vecs = model(batch["gloss_tensor"].to(args.device))
            for id, vec in zip(batch["id"], vecs):
                predictions.append({
                    "id": id,
                    args.target_arch: pred.view(-1).cpu().tolist()
                })
    with open(args.pred_file, "w") as ostr:
        json.dump(test_dataset, ostr)


def main(args):
    if args.do_train:
        utils.display("Performing training")
        train(args)
    if args.do_pred:
        utils.display("Performing prediction")
        pred(args)

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
