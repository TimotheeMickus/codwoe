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
            items = json.load(istr)
    except:
        raise ValueError(f"File \"{filename}\": could not open, submission will fail.")
    else:
        #expected_keys = {"id", "gloss"}
        for item in items:
            #keys_not_found = expected_keys - set(item.keys())
            if "id" not in item:
                raise ValueError(f"File \"{filename}\": one or more items do not contain an id, submission will fail.")
        ids = sorted([item["id"] for item in items])
        ids = [i.split('.') for i in ids]
        langs = {i[0] for i in ids}
        if len(langs) != 1:
            raise ValueError(f"File \"{filename}\": ids do not identify a unique language, submission will fail.")
        tracks = {i[-2] for i in ids}
        if len(tracks) != 1:
            raise ValueError(f"File \"{filename}\": ids do not identify a unique track, submission will fail.")
        track = next(iter(tracks))
        if track not in ("revdict", "defmod"):
            raise ValueError(f"File \"{filename}\": unknown track identified {track}, submission will fail.")
        lang = next(iter(langs))
        if lang not in ("en", "es", "fr", "it", "ru"):
            raise ValueError(f"File \"{filename}\": unknown language {lang}, submission will fail.")
        serials = list(sorted({int(i[-1]) for i in ids}))
        if serials != list(range(1, len(ids)+1)):
            raise ValueError(f"File \"{filename}\": ids do not identify all items in dataset, submission will fail.")
        if track == "revdict":
            vec_archs = set(item[0].keys()) - {"id", "gloss", "word", "pos", "concrete", "example", "f_rnk", "counts", "polysemous"}
            if len(vec_archs) == 0:
                raise ValueError(f"File \"{filename}\": no vector architecture was found, revdict submission will fail.")
            for item in item:
                if not all(v in item for v in vec_archs):
                    raise ValueError(f"File \"{filename}\": some items do not contain all the expected vectors, revdict submission will fail.")
            if len(vec_archs - {"sgns", "char", "electra"}):
                raise ValueError(f"File \"{filename}\": unknown vector architecture(s), revdict submission will fail.")
        if track == "defmod" and any("gloss" not in i for i in items):
            raise ValueError(f"File \"{filename}\": some items do not contain a gloss, defmod submission will fail.")

        ok_message = f'File "{filename}": no problems were identified.\n' + \
            f'The submission will be understood as follows:\n' + \
            f'\tSubmission on track {track} for language {lang}, {len(ids)} predictions.\n'
        if track == "revdict":
            ok_message += f'\tSubmission predicts these embeddings: {", ".join(vec_archs)}.'
        else:
            vec_archs = None
        utils.display(ok_message)
        CheckSummary = collections.namedtuple("CheckSummary", ["filename", "track", "lang", "vec_archs"])
        return CheckSummary(filename, track, lang, vec_archs)


if __name__ == "__main__":
    main(get_parser().parse_args().submission_file)
