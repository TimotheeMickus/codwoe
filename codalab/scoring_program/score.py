import argparse
import collections
import json
import pathlib

from nltk.translate.bleu_score import sentence_bleu as bleu
from nltk import word_tokenize as tokenize

import torch
import torch.nn.functional as F

import utils, check_output


def get_parser(parser=argparse.ArgumentParser(description="score a submission")):
    parser.add_argument("submission_file", type=pathlib.Path, help="file to check")
    parser.add_argument("--reference_files_dir", type=pathlib.Path, help="directory containing all reference files", default=pathlib.Path("data"))
    parser.add_argument("--output_file", type=pathlib.Path, help="default path to print output", default=pathlib.Path("scores.txt"))
    return parser

def eval_defmod(args, summary):
    # 1. read contents
    ## define accumulators for lemma-level BLEU and MoverScore
    reference_lemma_groups = collections.defaultdict(list)
    all_preds, all_tgts = [], []
    ## reading data files
    with open(args.submission_file, "r") as fp:
        submission = sorted(json.load(fp), key=lambda r: r["id"])
    with open(args.reference_file, "r") as fp:
        reference = sorted(json.load(fp), key=lambda r: r["id"])

    # 2. compute scores
    ## compute sense-level BLEU
    for sub, ref in zip(submission, reference):
        assert (sub["word"], sub["pos"]) == (ref["word"], ref["pos"]), "Mismatch in submission and reference files!"
        all_preds.append(sub["gloss"])
        all_tgts.append(ref["gloss"])
        sub["gloss"] = tokenize(sub["gloss"])
        ref["gloss"] = tokenize(ref["gloss"])
        sub["sense-BLEU"] = bleu([sub["gloss"]], ref["gloss"])
        reference_lemma_groups[(ref["word"], ref["pos"])].append(ref["gloss"])
    ## compute lemma-level BLEU
    for sub in submission:
        sub["lemma-BLEU"] = max(
            bleu([sub["gloss"]], g)
            for g in reference_lemma_groups[(sub["word"], sub["pos"])]
        )
    lemma_bleu_average = utils.average(s["lemma-BLEU"] for s in submission)
    sense_bleu_average = utils.average(s["sense-BLEU"] for s in submission)
    ## compute MoverScore
    moverscore_average = utils.mover_corpus_score(all_preds, [all_tgts])
    # 3. display results
    # utils.display(f"Submission {args.submission_file}, " + \
    #     f"MoverScore={moverscore_average} "+ \
    #     f"Lemma BLEU={lemma_bleu_average} "+ \
    #     f"Sense BLEU={sense_bleu_average}.")
    with open(args.output_file, "w") as ostr:
        print(f"MoverScore_{summary.lang}:{moverscore_average}", file=ostr)
        print(f"BLEU_lemma_{summary.lang}:{lemma_bleu_average}", file=ostr)
        print(f"BLEU_sense_{summary.lang}:{sense_bleu_average}", file=ostr)
    return args.submission_file, moverscore_average, lemma_bleu_average, sense_bleu_average

def rank_cosine(preds, targets):
    assocs = preds @ F.normalize(targets).T
    refs = torch.diagonal(assocs, 0).unsqueeze(1)
    ranks = (assocs >= refs).sum(1).float().mean().item()
    return ranks

def eval_revdict(args, summary):
    # 1. read contents
    ## read data files
    with open(args.submission_file, "r") as fp:
        submission = sorted(json.load(fp), key=lambda r: r["id"])
    with open(args.reference_file, "r") as fp:
        reference = sorted(json.load(fp), key=lambda r: r["id"])
    vec_archs = sorted(set(submission[0].keys()) - {"id", "gloss", "word", "pos", "concrete", "example", "f_rnk", "counts", "polysemous"})
    ## define accumulators for rank-cosine
    all_preds = collections.defaultdict(list)
    all_refs = collections.defaultdict(list)

    ## retrieve vectors
    for sub, ref in zip(submission, reference):
        assert (sub["pos"], sub["gloss"]) == (ref["pos"], ref["gloss"]), "Mismatch in submission and reference files!"
        for arch in vec_archs:
            all_preds[arch].append(sub[arch])
            all_refs[arch].append(ref[arch])


    torch.autograd.set_grad_enabled(False)
    all_preds = {
        arch:torch.tensor(all_preds[arch])
        for arch in vec_archs
    }
    all_refs = {
        arch:torch.tensor(all_refs[arch])
        for arch in vec_archs
    }

    # 2. compute scores
    MSE_scores = {
        arch: F.mse_loss(all_preds[arch], all_refs[arch]).item()
        for arch in vec_archs
    }
    cos_scores = {
        arch: F.cosine_similarity(all_preds[arch], all_refs[arch]).mean().item()
        for arch in vec_archs
    }
    rnk_scores = {
        arch:rank_cosine(all_preds[arch], all_refs[arch])
        for arch in vec_archs
    }
    # 3. display results
    # utils.display(f"Submission {args.submission_file}, \n\tMSE: " + \
    #     ", ".join(f"{a}={MSE_scores[a]}" for a in vec_archs) + \
    #     ", \n\tcosine: " + \
    #     ", ".join(f"{a}={cos_scores[a]}" for a in vec_archs) + \
    #     ", \n\tcosine ranks: " + \
    #     ", ".join(f"{a}={rnk_scores[a]}" for a in vec_archs) + \
    #     "."
    # )
    # all_archs = sorted(set(reference[0].keys()) - {"id", "gloss", "word", "pos"})
    with open(args.output_file, "w") as ostr:
        for arch in vec_archs:
            print(f"MSE_{summary.lang}_{arch}:{MSE_scores[arch]}", file=ostr)
            print(f"cos_{summary.lang}_{arch}:{cos_scores[arch]}", file=ostr)
            print(f"rnk_{summary.lang}_{arch}:{rnk_scores[arch]}", file=ostr)
    return (
        args.submission_file,
        *[MSE_scores.get(a, None) for a in vec_archs],
        *[cos_scores.get(a, None) for a in vec_archs]
    )


def main(args):
    if args.submission_file.is_dir():
        files = list(args.submission_file.glob("*.json"))
        assert len(files) == 1, "invalid submission: should contain exactly one JSON file."
        args.submission_file = files[0]
    if args.output_file.is_dir():
        args.output_file = args.output_file / "scores.txt"
    summary = check_output.main(args.submission_file)

    args.reference_file = args.reference_files_dir / f"{summary.lang}.{summary.track}.test.json"
    eval_func = eval_revdict if summary.track == "revdict" else eval_defmod
    return eval_func(args, summary)

if __name__ == "__main__":
    main(get_parser().parse_args())
