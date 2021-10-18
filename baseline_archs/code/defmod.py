import argparse
import itertools
import json
import logging
import pathlib
import pprint
import secrets

import skopt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import tqdm

import data
import models

logger = logging.getLogger(pathlib.Path(__file__).name)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(tqdm.tqdm)
handler.terminator = ""
handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(handler)


def get_parser(
    parser=argparse.ArgumentParser(description="run a definition modeling baseline"),
):
    parser.add_argument(
        "--do_htune",
        action="store_true",
        help="whether to perform hyperparameter tuning",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="whether to train a model from scratch"
    )
    parser.add_argument(
        "--do_pred", action="store_true", help="whether to produce predictions"
    )
    parser.add_argument(
        "--train_file", type=pathlib.Path, help="path to the train file"
    )
    parser.add_argument("--dev_file", type=pathlib.Path, help="path to the dev file")
    parser.add_argument("--test_file", type=pathlib.Path, help="path to the test file")
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cpu"),
        help="path to the train file",
    )
    parser.add_argument(
        "--source_arch",
        type=str,
        default="sgns",
        choices=("sgns", "char", "electra"),
        help="embedding architecture to use as source",
    )
    parser.add_argument(
        "--summary_logdir",
        type=pathlib.Path,
        default=pathlib.Path("logs") / "defmod-baseline",
        help="write logs for future analysis",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        default=pathlib.Path("models") / "defmod-baseline",
        help="where to save model & vocab",
    )
    parser.add_argument(
        "--spm_model_path",
        type=pathlib.Path,
        default=None,
        help="use sentencepiece model, if required train and save it here",
    )
    parser.add_argument(
        "--pred_file",
        type=pathlib.Path,
        default=pathlib.Path("defmod-baseline-preds.json"),
        help="where to save predictions",
    )
    return parser


def get_search_space():
    """get hyperparmeters to optimize for"""
    search_space = [
        skopt.space.Real(1e-8, 1.0, "log-uniform", name="learning_rate"),
        skopt.space.Real(0.0, 1.0, "uniform", name="weight_decay"),
        skopt.space.Real(0.9, 1.0 - 1e-8, "log-uniform", name="beta_a"),
        skopt.space.Real(0.9, 1.0 - 1e-8, "log-uniform", name="beta_b"),
        skopt.space.Real(0.0, 0.9, "uniform", name="dropout"),
        skopt.space.Real(0.0, 1.0, "uniform", name="warmup_len"),
        skopt.space.Real(0.0, 1.0 - 1e-8, "uniform", name="label_smoothing"),
        skopt.space.Integer(1, 100, "log-uniform", name="batch_accum"),
        skopt.space.Integer(0, 5, "uniform", name="n_head_pow"),
        skopt.space.Integer(1, 6, "uniform", name="n_layers"),
    ]
    return search_space


def train(
    train_file,
    dev_file,
    source_arch="sgns",
    summary_logdir=pathlib.Path("logs") / "defmod-htune",
    save_dir=pathlib.Path("models") / "defmod-baseline",
    device="cuda:0",
    spm_model_path=None,
    epochs=100,
    learning_rate=1e-4,
    beta1=0.9,
    beta2=0.999,
    weight_decay=1e-6,
    patience=5,
    batch_accum=1,
    dropout=0.3,
    warmup_len=0.1,
    label_smoothing=0.1,
    n_head=4,
    n_layers=4,
):
    assert train_file is not None, "Missing dataset for training"
    assert dev_file is not None, "Missing dataset for development"

    # 1. get data, vocabulary, summary writer
    logger.debug("Preloading data")
    save_dir = save_dir / source_arch
    save_dir.mkdir(parents=True, exist_ok=True)
    ## make datasets
    train_dataset = data.get_train_dataset(train_file, spm_model_path, save_dir)
    dev_dataset = data.get_dev_dataset(
        dev_file, spm_model_path, save_dir, train_dataset
    )
    ## assert they correspond to the task
    assert train_dataset.has_gloss, "Training dataset contains no gloss."
    if source_arch == "electra":
        assert train_dataset.has_electra, "Training datatset contains no vector."
    else:
        assert train_dataset.has_vecs, "Training datatset contains no vector."
    assert dev_dataset.has_gloss, "Development dataset contains no gloss."
    if source_arch == "electra":
        assert dev_dataset.has_electra, "Development dataset contains no vector."
    else:
        assert dev_dataset.has_vecs, "Development dataset contains no vector."
    ## make dataloader
    train_dataloader = data.get_dataloader(train_dataset)
    dev_dataloader = data.get_dataloader(dev_dataset, shuffle=False)
    ## make summary writer
    summary_writer = SummaryWriter(summary_logdir)
    train_step = itertools.count()  # to keep track of the training steps for logging

    # 2. construct model
    logger.debug("Setting up training environment")
    model = models.DefmodModel(
        dev_dataset.vocab, n_head=n_head, n_layers=n_layers, dropout=dropout
    )
    model = model.to(device)
    model.train()

    # 3. declare optimizer & criterion
    ## Hyperparams
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )
    xent_criterion = nn.CrossEntropyLoss(ignore_index=model.padding_idx)
    if label_smoothing > 0.0:
        smooth_criterion = models.LabelSmoothingCrossEntropy(
            ignore_index=model.padding_idx, epsilon=label_smoothing
        )
    else:
        smooth_criterion = xent_criterion

    vec_tensor_key = f"{source_arch}_tensor"
    best_xent = float("inf")
    strikes = 0

    # 4. train model
    epochs_range = tqdm.trange(epochs, desc="Epochs")
    total_steps = (len(train_dataloader) * epochs) // batch_accum
    scheduler = models.get_schedule(
        optimizer, round(total_steps * warmup_len), total_steps
    )
    for epoch in epochs_range:
        ## train loop
        pbar = tqdm.tqdm(
            desc=f"Train {epoch}", total=len(train_dataset), disable=None, leave=False
        )
        optimizer.zero_grad()
        for i, batch in enumerate(train_dataloader):
            vec = batch[vec_tensor_key].to(device)
            gls = batch["gloss_tensor"].to(device)
            pred = model(vec, gls[:-1])
            loss = smooth_criterion(pred.view(-1, pred.size(-1)), gls.view(-1))
            loss.backward()
            grad_remains = True
            step = next(train_step)
            if i % batch_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                grad_remains = False
                summary_writer.add_scalar(
                    "defmod-train/lr", scheduler.get_last_lr()[0], step
                )
            # keep track of the train loss for this step
            with torch.no_grad():
                tokens = gls != model.padding_idx
                acc = (
                    ((pred.argmax(-1) == gls) & tokens).float().sum() / tokens.sum()
                ).item()
                xent_unsmoothed = xent_criterion(
                    pred.view(-1, pred.size(-1)), gls.view(-1)
                )
                summary_writer.add_scalar("defmod-train/xent_smooth", loss.item(), step)
                summary_writer.add_scalar("defmod-train/xent", xent_unsmoothed, step)
                summary_writer.add_scalar("defmod-train/acc", acc, step)
            pbar.update(vec.size(0))
        if grad_remains:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        pbar.close()
        ## eval loop
        model.eval()
        with torch.no_grad():
            sum_dev_loss = 0.0
            sum_acc = 0
            ntoks = 0
            pbar = tqdm.tqdm(
                desc=f"Eval {epoch}",
                total=len(dev_dataset),
                disable=None,
                leave=False,
            )
            for batch in dev_dataloader:
                vec = batch[vec_tensor_key].to(device)
                gls = batch["gloss_tensor"].to(device)
                pred = model(vec, gls[:-1])
                sum_dev_loss += F.cross_entropy(
                    pred.view(-1, pred.size(-1)),
                    gls.view(-1),
                    reduction="sum",
                    ignore_index=model.padding_idx,
                ).item()
                tokens = gls != model.padding_idx
                ntoks += tokens.sum().item()
                sum_acc += ((pred.argmax(-1) == gls) & tokens).sum().item()
                pbar.update(vec.size(0))

            # keep track of the average loss & acc on dev set for this epoch
            new_xent = sum_dev_loss / ntoks
            summary_writer.add_scalar("defmod-dev/xent", new_xent, epoch)
            summary_writer.add_scalar("defmod-dev/acc", sum_acc / ntoks, epoch)
            pbar.close()
            if new_xent < (best_xent * 0.999):
                logger.debug(
                    f"Epoch {epoch}, new best loss: {new_xent:.4f} < {best_xent:.4f}"
                    + f" (x 0.999 = {best_xent * 0.999:.4f})"
                )
                best_xent = new_xent
                strikes = 0
            else:
                strikes += 1
            # check result if better
            if not (save_dir / "best_scores.txt").is_file():
                overall_best_xent = float("inf")
            else:
                with open(save_dir / "best_scores.txt", "r") as score_file:
                    overall_best_xent = float(score_file.read())
            # save result if better
            if new_xent < overall_best_xent:
                logger.debug(
                    f"Epoch {epoch}, new overall best loss: {new_xent:.4f} < {overall_best_xent:.4f}"
                )
                model.save(save_dir / "model.pt")
                with open(save_dir / "hparams.json", "w") as json_file:
                    hparams = {
                        "learning_rate": learning_rate,
                        "beta1": beta1,
                        "beta2": beta2,
                        "weight_decay": weight_decay,
                    }
                    json.dump(hparams, json_file, indent=2)
                with open(save_dir / "best_scores.txt", "w") as score_file:
                    print(new_xent, file=score_file)

        if strikes >= patience:
            logger.debug("Stopping early.")
            epochs_range.close()
            break
        model.train()
    # return loss for gp minimize
    return best_xent


def pred(args):
    assert args.test_file is not None, "Missing dataset for test"
    # 1. retrieve vocab, dataset, model
    model = models.DefmodModel.load(args.save_dir / "model.pt")
    train_vocab = data.JSONDataset.load(args.save_dir / "train_dataset.pt").vocab
    test_dataset = data.JSONDataset(
        args.test_file, vocab=train_vocab, freeze_vocab=True, maxlen=model.maxlen, spm_model_name=args.spm_model_path
    )
    test_dataloader = data.get_dataloader(test_dataset, shuffle=False, batch_size=1)
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
            sequence = model.pred(batch[vec_tensor_key].to(args.device), decode_fn=test_dataset.decode, verbose=False)
            for id, gloss in zip(batch["id"], test_dataset.decode(sequence)):
                predictions.append({"id": id, "gloss": gloss})
            pbar.update(batch[vec_tensor_key].size(0))
        pbar.close()
    # 3. dump predictions
    with open(args.pred_file, "w") as ostr:
        json.dump(predictions, ostr)


def main(args):
    assert not (args.do_train and args.do_htune), "Conflicting options"
    if args.do_train:
        logger.debug("Performing defmod training")
        train(
            args.train_file,
            args.dev_file,
            args.source_arch,
            args.summary_logdir,
            args.save_dir,
            args.device,
        )
    elif args.do_htune:
        logger.debug("Performing defmod hyperparameter tuning")
        search_space = get_search_space()

        @skopt.utils.use_named_args(search_space)
        def gp_train(**hparams):
            logger.debug(f"Hyperparams sampled:\n{pprint.pformat(hparams)}")
            best_loss = train(
                train_file=args.train_file,
                dev_file=args.dev_file,
                source_arch=args.source_arch,
                summary_logdir=args.summary_logdir
                / args.source_arch
                / secrets.token_urlsafe(8),
                save_dir=args.save_dir,
                device=args.device,
                spm_model_path=args.spm_model_path,
                learning_rate=hparams["learning_rate"],
                beta1=min(hparams["beta_a"], hparams["beta_b"]),
                beta2=max(hparams["beta_a"], hparams["beta_b"]),
                weight_decay=hparams["weight_decay"],
                batch_accum=hparams["batch_accum"],
                warmup_len=hparams["warmup_len"],
                label_smoothing=hparams["label_smoothing"],
                n_head=2 ** hparams["n_head_pow"],
                n_layers=hparams["n_layers"],
            )
            return best_loss

        result = skopt.gp_minimize(gp_train, search_space)
        args.save_dir = args.save_dir / args.source_arch
        skopt.dump(result, args.save_dir / "results.pkl", store_objective=False)

    if args.do_pred:
        logger.debug("Performing defmod prediction")
        pred(args)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
