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
    parser.add_argument("--target-arch", type=str, default="w2v", help="embedding architecture to use as target")
    parser.add_argument("--summary-logdir", type=pathlib.Path, default=pathlib.Path("logs") / f"revdict-baseline", help="write logs for future analysis")
    parser.add_argument("--save-dir", type=pathlib.Path, default=pathlib.Path("models") / f"revdict-baseline", help="where to save model & vocab")
    parser.add_argument("--pred-file", type=pathlib.Path, default=pathlib.Path("revdict-baseline-preds.json"), help="where to save predictions")
    return parser


def train(args):
    assert args.train_file is not None, "Missing dataset for training"
    # 1. get data, vocabulary, summary writer
    utils.display("Preloading training data")
    train_dataset, vocab = utils.read_dataset(args.train_file, device=args.device)
    if args.dev_file:
        dev_dataset, _ = utils.read_dataset(args.dev_file, vocab=vocab, device=args.device)
    # save vocab for future work
    args.save_dir.mkdir(parents=True, exist_ok=True)
    with open(args.save_dir / "vocab.json", "w") as ostr:
        json.dump(vocab, ostr)
    summary_writer = SummaryWriter(args.summary_logdir)
    train_step = itertools.count() # to keep track of the training steps for logging

    # 2. construct model
    ## Hyperparams
    D_MODEL, N_HEAD, N_LAYERS, DROPOUT = 256, 4, 6, 0.1
    ## declare model
    encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD, dropout=DROPOUT, dim_feedforward=D_MODEL*2)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
    model = nn.Sequential(
        nn.Embedding(len(vocab), D_MODEL, padding_idx=vocab[utils.PAD_token]),
        utils.PositionalEncoding(D_MODEL, dropout=DROPOUT),
        transformer_encoder,
        utils.SumLayer(dim=0)
    ).to(args.device)
    ## initialize weights
    for name, param in model.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
        else: # gain parameters of the layer norm
            nn.init.ones_(param)
    model.train()

    # 3. declare optimizer & criterion
    ## Hyperparams
    EPOCHS, LEARNING_RATE, BETA1, BETA2, WEIGHT_DECAY = 10, 1.e-4, .9, .999, 1.e-5
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # 4. train model
    for epoch in tqdm.trange(EPOCHS, desc="Epoch"):
        ## train loop
        for example in tqdm.tqdm(train_dataset, desc="Train", leave=False):
            optimizer.zero_grad()
            src = example["tensors"]["gloss"].unsqueeze(1)
            tgt = example["tensors"][args.target_arch].unsqueeze(0)
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
                for example in tqdm.tqdm(dev_dataset, desc="Eval.", leave=False):
                    src = example["tensors"]["gloss"].unsqueeze(1)
                    tgt = example["tensors"][args.target_arch].unsqueeze(0)
                    pred = model(src)
                    sum_dev_loss += criterion(pred, tgt).item()
                # keep track of the average loss on dev set for this epoch
                summary_writer.add_scalar("dev/loss", sum_dev_loss / len(dev_dataset), epoch)
            model.train()

    # 5. save result
    torch.save(model, args.save_dir / "model.pt")

def pred(args):
    assert args.test_file is not None, "Missing dataset for test"
    # 1. retrieve vocab, dataset, model
    with open(args.save_dir / "vocab.json", "r") as istr:
        vocab = json.load(istr)
    test_dataset, _ = utils.read_dataset(args.test_file, vocab=vocab, device=args.device)
    model = torch.load(args.save_dir / "model.pt")
    model.eval()
    # 2. make predictions
    with torch.no_grad():
        for example in tqdm.tqdm(test_dataset, desc="Pred."):
            src = example["tensors"]["gloss"].unsqueeze(1)
            pred = model(src)
            example[args.target_arch] = pred.view(-1).tolist()
            del example["tensors"]
    with open(args.pred_file, "w") as ostr:
        json.dump(test_dataset, ostr, indent=2)


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
