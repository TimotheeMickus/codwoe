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
    parser=argparse.ArgumentParser(
        description="Run a reverse dictionary baseline.\nThe task consists in reconstructing an embedding from the glosses listed in the datasets"
    ),
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
        "--target_arch",
        type=str,
        default="sgns",
        choices=("sgns", "char", "electra"),
        help="embedding architecture to use as target",
    )
    parser.add_argument(
        "--summary_logdir",
        type=pathlib.Path,
        default=pathlib.Path("logs") / f"revdict-baseline",
        help="write logs for future analysis",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        default=pathlib.Path("models") / f"revdict-baseline",
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
        default=pathlib.Path("revdict-baseline-preds.json"),
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
        skopt.space.Integer(1, 100, "log-uniform", name="batch_accum"),
        skopt.space.Integer(0, 5, "uniform", name="n_head_pow"),
        skopt.space.Integer(1, 6, "uniform", name="n_layers"),
    ]
    return search_space


def train(
    train_file,
    dev_file,
    target_arch="sgns",
    summary_logdir=pathlib.Path("logs") / "revdict-htune",
    save_dir=pathlib.Path("models") / "revdict-baseline",
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
    n_head=4,
    n_layers=4,
):
    assert train_file is not None, "Missing dataset for training"
    assert dev_file is not None, "Missing dataset for development"
    # 1. get data, vocabulary, summary writer
    logger.debug("Preloading data")
    save_dir = save_dir / target_arch
    save_dir.mkdir(parents=True, exist_ok=True)
    ## make datasets
    train_dataset = data.get_train_dataset(train_file, spm_model_path, save_dir)
    dev_dataset = data.get_dev_dataset(
        dev_file, spm_model_path, save_dir, train_dataset
    )

    ## assert they correspond to the task
    assert train_dataset.has_gloss, "Training dataset contains no gloss."
    if target_arch == "electra":
        assert train_dataset.has_electra, "Training datatset contains no vector."
    else:
        assert train_dataset.has_vecs, "Training datatset contains no vector."
    assert dev_dataset.has_gloss, "Development dataset contains no gloss."
    if target_arch == "electra":
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
    ## Hyperparams
    logger.debug("Setting up training environment")
    model = models.RevdictModel(
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
    criterion = nn.MSELoss()

    vec_tensor_key = f"{target_arch}_tensor"
    best_mse = float("inf")
    strikes = 0

    # 4. train model
    epochs_range = tqdm.trange(epochs, desc="Epochs")
    total_steps = (len(train_dataloader) * epochs) // batch_accum
    scheduler = models.get_schedule(
        optimizer, round(total_steps * warmup_len), total_steps
    )

    # 4. train model
    for epoch in epochs_range:
        ## train loop
        pbar = tqdm.tqdm(
            desc=f"Train {epoch}", total=len(train_dataset), disable=None, leave=False
        )
        optimizer.zero_grad()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            gls = batch["gloss_tensor"].to(device)
            vec = batch[vec_tensor_key].to(device)
            pred = model(gls)
            loss = criterion(pred, vec)
            loss.backward()
            grad_remains = True
            step = next(train_step)
            if i % batch_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                grad_remains = False
                summary_writer.add_scalar(
                    "revdict-train/lr", scheduler.get_last_lr()[0], step
                )
            # keep track of the train loss for this step
            with torch.no_grad():
                cos_sim = F.cosine_similarity(pred, vec).mean().item()
                summary_writer.add_scalar("revdict-train/cos", cos_sim, step)
                summary_writer.add_scalar("revdict-train/mse", loss.item(), step)
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
            sum_cosine = 0.0
            pbar = tqdm.tqdm(
                desc=f"Eval {epoch}",
                total=len(dev_dataset),
                disable=None,
                leave=False,
            )
            for batch in dev_dataloader:
                gls = batch["gloss_tensor"].to(device)
                vec = batch[vec_tensor_key].to(device)
                pred = model(gls)
                sum_dev_loss += (
                    F.mse_loss(pred, vec, reduction="none").mean(1).sum().item()
                )
                sum_cosine += F.cosine_similarity(pred, vec).sum().item()
                pbar.update(vec.size(0))
            # keep track of the average loss on dev set for this epoch
            new_mse = sum_dev_loss / len(dev_dataset)
            summary_writer.add_scalar(
                "revdict-dev/cos", sum_cosine / len(dev_dataset), epoch
            )
            summary_writer.add_scalar("revdict-dev/mse", new_mse, epoch)
            pbar.close()
            if new_mse < (best_mse * 0.999):
                logger.debug(
                    f"Epoch {epoch}, new best loss: {new_mse:.4f} < {best_mse:.4f}"
                    + f" (x 0.999 = {best_mse * 0.999:.4f})"
                )
                best_mse = new_mse
                strikes = 0
            else:
                strikes += 1
            # check result if better
            if not (save_dir / "best_scores.txt").is_file():
                overall_best_mse = float("inf")
            else:
                with open(save_dir / "best_scores.txt", "r") as score_file:
                    overall_best_mse = float(score_file.read())
            # save result if better
            if new_mse < overall_best_mse:
                logger.debug(
                    f"Epoch {epoch}, new overall best loss: {new_mse:.4f} < {overall_best_mse:.4f}"
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
                    print(new_mse, file=score_file)
        if strikes >= patience:
            logger.debug("Stopping early.")
            epochs_range.close()
            break
        model.train()
    # return loss for gp minimize
    return best_mse


def pred(args):
    assert args.test_file is not None, "Missing dataset for test"
    # 1. retrieve vocab, dataset, model
    model = models.DefmodModel.load(args.save_dir / "model.pt")
    train_vocab = data.JSONDataset.load(args.save_dir / "train_dataset.pt").vocab
    test_dataset = data.JSONDataset(
        args.test_file, vocab=train_vocab, freeze_vocab=True, maxlen=model.maxlen
    )
    test_dataloader = data.get_dataloader(test_dataset, shuffle=False, batch_size=1024)
    model.eval()
    vec_tensor_key = f"{args.target_arch}_tensor"
    assert test_dataset.has_gloss, "File is not usable for the task"
    # 2. make predictions
    predictions = []
    with torch.no_grad():
        pbar = tqdm.tqdm(desc="Pred.", total=len(test_dataset))
        for batch in test_dataloader:
            vecs = model(batch["gloss_tensor"].to(args.device)).cpu()
            for id, vec in zip(batch["id"], vecs.unbind()):
                predictions.append(
                    {"id": id, args.target_arch: vec.view(-1).cpu().tolist()}
                )
            pbar.update(vecs.size(0))
        pbar.close()
    with open(args.pred_file, "w") as ostr:
        json.dump(predictions, ostr)


def main(args):
    assert not (args.do_train and args.do_htune), "Conflicting options"

    if args.do_train:
        logger.debug("Performing revdict training")
        train(
            args.train_file,
            args.dev_file,
            args.target_arch,
            args.summary_logdir,
            args.save_dir,
            args.device,
        )
    elif args.do_htune:
        logger.debug("Performing revdict hyperparameter tuning")
        search_space = get_search_space()

        @skopt.utils.use_named_args(search_space)
        def gp_train(**hparams):
            logger.debug(f"Hyperparams sampled:\n{pprint.pformat(hparams)}")
            best_loss = train(
                train_file=args.train_file,
                dev_file=args.dev_file,
                target_arch=args.target_arch,
                summary_logdir=args.summary_logdir
                / args.target_arch
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
                n_head=2 ** hparams["n_head_pow"],
                n_layers=hparams["n_layers"],
            )
            return best_loss

        result = skopt.gp_minimize(gp_train, search_space)
        args.save_dir = args.save_dir / args.target_arch
        skopt.dump(result, args.save_dir / "results.pkl", store_objective=False)

    if args.do_pred:
        logger.debug("Performing revdict prediction")
        pred(args)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
