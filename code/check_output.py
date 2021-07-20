import argparse
import collections
import pathlib
import json

import utils


def get_parser(parser=argparse.ArgumentParser(description="Verify the output format of a submission")):
    parser.add_argument("submission_file", type=pathlib.Path, help="file to check")
    return parser


def main(filename):
    try:
        with open(filename, "r") as istr:
            data = json.load(istr)
    except:
        raise ValueError(f"File \"{filename}\": could not open, submission will fail.")
    else:
        expected_keys = {"id", "gloss", "word", "pos"}
        vec_archs = set(data[0].keys()) - expected_keys
        if len(vec_archs) == 0:
            raise ValueError(f"File \"{filename}\": no vector architecture was found, submission will fail.")
        expected_keys = expected_keys | vec_archs
        for item in data:
            keys_not_found = expected_keys - set(item.keys())
            if keys_not_found:
                raise ValueError(f"File \"{filename}\": one or more items do not contain all the expected keys, submission will fail.")
        ids = sorted([item["id"] for item in data])
        ids = [i.split('.') for i in ids]
        langs = {i[0] for i in ids}
        if len(langs) != 1:
            raise ValueError(f"File \"{filename}\": ids do not identify a unique language, submission will fail.")
        tracks = {i[1] for i in ids}
        if len(tracks) != 1:
            raise ValueError(f"File \"{filename}\": ids do not identify a unique track, submission will fail.")
        serials = list(sorted({int(i[2]) for i in ids}))
        if serials != list(range(1, len(ids)+1)):
            raise ValueError(f"File \"{filename}\": ids do not identify all items in dataset, submission will fail.")
        track = next(iter(tracks))
        if track not in ("revdict", "defmod", "trial"):
            raise ValueError(f"File \"{filename}\": unknown track identified {track}, submission will fail.")
        lang = next(iter(langs))
        if lang not in ("en", "es", "fr", "it", "ru"):
            raise ValueError(f"File \"{filename}\": unknown language {lang}, submission will fail.")
        ok_message = f'File "{filename}": no problems were identified.\n' + \
            f'The submission will be understood as follows:\n' + \
            f'\tSubmission on track {track} for language {lang}, {len(ids)} predictions.\n'
        if track == "revdict":
            ok_message += f'\tSubmission predicts these embeddings: {", ".join(vec_archs)}.'
        else:
            vec_archs = None
        if track == "trial":
            track = "defmod"
        utils.display(ok_message)
        CheckSummary = collections.namedtuple("CheckSummary", ["filename", "track", "lang", "vec_archs"])
        return CheckSummary(filename, track, lang, vec_archs)


if __name__ == "__main__":
    main(get_parser().parse_args().submission_file)
