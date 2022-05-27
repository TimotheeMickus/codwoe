# Comparing Dictionaries and Word Embeddings

This is the repository for the SemEval 2022 Shared Task #1: Comparing
Dictionaries and Word Embeddings (CODWOE).

This repository currently contains: the configuration for the codalab
competition, a Docker image to reproduce the environment, a scorer, a
format-checker and baseline programs to help participants get started.

Participants may be interested in the script `codwoe_entrypoint.py`. It contains
a number of useful features, such as scoring submissions, a format checker and a
few simple baseline architectures. It is also the exact copy of what is used on
the codalab.

# Use this repository

The code and data of this shared task are provided CC-BY-SA.
Also consider citing the task paper (to appear):
```
@inproceedings{mickus-etal-2022-semeval,
    title={{SemEval-2022 Task 1}: CODWOE -- COmparing Dictionaries and WOrd Embeddings},
    author={Mickus, Timothee and Paperno, Denis and Constant, Mathieu and van Deemter, Kees},
    booktitle={Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)}
    year={2022},
    publisher={Association for Computational Linguistics},
}
```

<!-- We provide data using git LFS. Make sure to install the relevant software if you
wish to clone this repository: see the documentation
[here](https://git-lfs.github.com/). -->

**Datasets are no longer provided directly on this repository. The competition datasets are now available on this page: [https://codwoe.atilf.fr/](https://codwoe.atilf.fr/).**

# What is this task?
The CODWOE shared task invites you to compare two types of semantic
descriptions: dictionary glosses and word embedding representations. Are these
two types of representation equivalent? Can we generate one from the other? To
study this question, we propose two subtracks: a **definition modeling** track
(Noraset et al., 2017), where participants have to generate glosses from
vectors, and a **reverse dictionary** track (Hill et al., 2016), where
participants have to generate vectors from glosses.

These two tracks display a number of interesting characteristics. Definition
modeling is a vector-to-sequence task, the reverse dictionary task is a
sequence-to-vector task—and you know that kind of thing gets NLP people swearing
out loud. These tasks are also useful for explainable AI, since they involve
converting human-readable data into machine-readable data and back.

To get involved: check out the
[codalab competition](https://competitions.codalab.org/competitions/34022).
There is also a participants'
["semeval2022-dictionaries-and-word-embeddings" google group](mailto:semeval2022-dictionaries-and-word-embeddings@googlegroups.com),
as well as a [discord server](https://discord.gg/y8g6qXakNs).
You can reach us organizers through [this email](mailto:tmickus@atilf.fr); make
sure to mention SemEval in your email object.

# How hard is it?

## Official rankings

Below are the official rankings for the SemEval 2022 CODWOE Shared task.
More information about the submissions we received is available in this git (see the `rankings/` sub-directory).

### Definition Modeling track

Below are the results for the Definition Modeling track.

|  user / team  | Rank EN | Rank ES | Rank FR | Rank IT | Rank RU
|---------------|--------:|--------:|--------:|--------:|--------:
| Locchi        | 8       | 6       |         | 7       |
| LingJing      | 9       | 7       | 6       | 6       | 6
| BLCU-ICALL    | 3       | 2       | 3       | **1**   | 2
| IRB-NLP       | 2       | **1**   | **1**   | 5       | 5
| emukans       | 5       | 4       | 4       | 4       | 3
| guntis        | 6       |         |         |         |
| lukechan1231  | 7       | 5       | 5       | 3       | 4
| pzchen        | 4       | 3       | 2       | 2       | **1**   
| talent404     | **1**   |         |         |         |

### Reverse Dictionary track

Below are the results for the Reverse dictionary tracks.
There are separate rankings, based on which targets participants have submitted.


#### A. SGNS targets

| user / team      | Rank EN | Rank ES | Rank FR | Rank IT | Rank RU
|------------------|--------:|--------:|--------:|--------:|--------:
| Locchi           | 4       |         |         | 4       |
| BL.Research      | 5       | 5       | 4       | 6       | 4
| LingJing         | **1**   | 2       | 2       | 3       | **1**
| MMG              |         | 3       |         |         |
| chlrbgus321      | N/A     |         |         |         |
| IRB-NLP          | 3       | **1**   | **1**   | **1**   | 2
| pzchen           | 2       | 4       | 3       | 2       | 3
| the0ne           | 7       |         |         |         |
| JSI              | 8       | 7       | 6       | 7       | 6
| zhwa3087         | 6       | 6       | 5       | 5       | 5

#### B. ELECTRA targets

| user / team      | Rank EN | Rank FR | Rank RU
|------------------|--------:|--------:|--------:
| Locchi           | 3       |         |
| BL.Research      | 2       | 2       | 4
| LingJing         | 4       | 4       | 2
| IRB-NLP          | 5       | 3       | 3
| pzchen           | **1**   | **1**   | **1**
| the0ne           | 6       |         |


#### C. Char-based targets

| user / team      | Rank EN | Rank ES | Rank FR | Rank IT | Rank RU
|------------------|--------:|--------:|--------:|--------:|--------:
| Locchi           | **1**   |         |         | 4       |
| BL.Research      | 2       | 2       | 2       | 3       | 4
| LingJing         | 7       | 5       | 5       | 6       | 5
| IRB-NLP          | 4       | 3       | 4       | 2       | 2
| pzchen           | 3       | **1**   | **1**   | **1**   | **1**
| the0ne           | 5       |         |         |         |
| zhwa3087         | 6       | 4       | 3       | 5       | 3



## Baseline results
Here are baseline results on the development set for the two tracks.
We used the code described in `code/baseline_archs` to generate these scores.

For the Reverse Dictionary track results, rows will correspond to different targets.
On the other hand, rows of the Definition Modeling table below correspond to different inputs to the system.
Scores were computed using the scoring script provided in this git (`code/score.py`).

### Reverse Dictionary track

|            | MSE     | Cosine  | Ranking
|------------|--------:|--------:|--------:
| en SGNS    | 0.91092 | 0.15132 | 0.49030
| en char    | 0.14776 | 0.79006 | 0.50218
| en electra | 1.41287 | 0.84283 | 0.49849
| es SGNS    | 0.92996 | 0.20406 | 0.49912
| es char    | 0.56952 | 0.80634 | 0.49778
| fr SGNS    | 1.14050 | 0.19774 | 0.49052
| fr char    | 0.39480 | 0.75852 | 0.49945
| fr electra | 1.15348 | 0.85629 | 0.49784
| it SGNS    | 1.12536 | 0.20430 | 0.47692
| it char    | 0.36309 | 0.72732 | 0.49663
| ru SGNS    | 0.57683 | 0.25316 | 0.49008
| ru char    | 0.13498 | 0.82624 | 0.49451
| ru electra | 0.87358 | 0.72086 | 0.49120


### Definition Modeling track

|            | Sense-BLEU | Lemma-BLEU | MoverScore
|------------|-----------:|-----------:|-----------:
| en SGNS    | 0.03048    | 0.04062    | 0.08307
| en char    | 0.02630    | 0.03359    | 0.04531
| en electra | 0.03155    | 0.04155    | 0.06732
| es SGNS    | 0.03528    | 0.05273    | 0.06685
| es char    | 0.03291    | 0.04712    | 0.06112
| fr SGNS    | 0.02983    | 0.04134    | 0.04036
| fr char    | 0.02913    | 0.03985    | 0.01935
| fr electra | 0.03061    | 0.03954    | 0.03855
| it SGNS    | 0.04759    | 0.06910    | 0.10154
| it char    | 0.02532    | 0.03522    | 0.04068
| ru SGNS    | 0.03805    | 0.05121    | 0.11559
| ru char    | 0.02324    | 0.03238    | 0.07145
| ru electra | 0.02987    | 0.03782    | 0.10382

# Using this repository
To install the exact environment used for our scripts, see the
`requirements.txt` file which lists the library we used. Do
note that the exact installation in the competition underwent supplementary
tweaks: in particular, we patch the moverscore library to have it run on CPU.

Another possibility is to use the dockerfile written for the codalab
competition. You can also pull this docker image from dockerhub:
[`linguistickus/codwoe`](https://hub.docker.com/r/linguistickus/codwoe). This
Docker image doesn't contain the code, so you will also need to clone the
repository within it; but this image will also contain our tweaks.

Code useful to participants is stored in the `code/` directory.
To see options a simple baseline on the definition modeling track, use:
```sh
$ python3 code/codwoe_entrypoint.py defmod --help
```
To see options for a simple baseline on the reverse dictionary track, use:
```sh
$ python3 code/codwoe_entrypoint.py revdict --help
```
To verify the format of a submission, run:
```sh
$ python3 code/codwoe_entrypoint.py check-format $PATH_TO_SUBMISSION_FILE
```
To score a submission, use  
```sh
$ python3 code/codwoe_entrypoint.py score $PATH_TO_SUBMISSION_FILE --reference_files_dir $PATH_TO_DATA_DIR
```
Note that this requires the gold files, not available at the start of the
competition.

Other useful files to look at include `code/models.py`, where our baseline
architectures are defined, and `code/data.py`, which shows how to use the JSON
datasets with the PyTorch dataset API.

# Using the datasets

**Datasets are no longer provided directly on this repository. The competition datasets are now available on this page: [https://codwoe.atilf.fr/](https://codwoe.atilf.fr/).**

This section details the structure of the JSON dataset file we provide. More information is available on the competition website: [link](https://competitions.codalab.org/competitions/34022#participate-get_data).

### Brief Overview

As an overview, the expected usage of the datasets is as follow:
 + In the Definition Modeling track, we expect participants to use the embeddings ("char", "sgns", "electra") to generate the associated definition ("gloss").
 + In the Reverse Dictionary track, we expect participants to use the definition ("gloss") to generate any of the associated embeddings ("char", "sgns", "electra").


### Dataset files structure

Each dataset file correspond to a data split (trial/train/dev/test) for one of the languages.

Dataset files are in the JSON format. A dataset file contains a list of examples. Each example is a JSON dictionary, containing the following keys:
 + "id",
 + "gloss"
 + "sgns"
 + "char"

The English, French and Russian dictionary also contain an "electra" key.

As a concrete instance, here is an example from the English training dataset:
```json
    {
        "id": "en.train.2",
        "gloss": "A vocal genre in Hindustani classical music",
        "sgns": [
            -0.0602365807,
           ...
        ],
        "char": [
            -0.3631578386,
           ...
        ],
        "electra": [
            -1.3904430866,
           ...
     ]
    },
```

### Description of contents

The value associated to "id" tracks the language, data split and unique identifier for this example.

The value associated to the "gloss" key is a definition, as you would find in a classical dictionary. It is to be used either the target in the Definition Modeling track, or  asthe source in the Reverse Dictionary track.

All other keys ("char", "sgns", "electra") correspond to embeddings, and the associated values are arrays of floats representing the components. They all can serve as targets for the Reverse Dictionary track.
 + "char" corresponds to character-based embeddings, computed using an auto-encoder on the spelling of a word.
 + "sgns" corresponds to skip-gram with negative sampling embeddings (aka. word2vec)
 + "electra" corresponds to Transformer-based contextualized embeddings.


### Using the dataset files

Given that the data is in JSON format, it is straightforward to load it in python:

```python
import json
with open(PATH_TO_DATASET, "r") as file_handler:
    dataset = json.load(file_handler)
```

A more complete example for pytorch is available in the git repository (see here: [link](https://git.atilf.fr/tmickus/codwoe/-/blob/master/code/data.py#L18)).

### Expected output format

During the evaluation phase, we will expect submissions to reconstruct the same JSON format.

The test JSON files for input will be separate for each track. They will contain the "id" key, and either the "gloss" key (in the reverse dictionary track) or the embedding keys ("char" and "sgns" keys, and "electra" "key" in EN/FR/RU, in the definition modeling track).

In the definition modeling track, participants should construct JSON files that contain at least the two following keys:
 + the original "id"
 + their generated "gloss"

In the reverse dictionary, participants should construct JSON files that contain at least the two following keys:
 + the original "id",
 + any of the valid embeddings ("char", "sgns", or "electra" key in EN/FR/RU)

Other keys can be added. More details concerning the evaluation procedure are available here: [link](https://competitions.codalab.org/competitions/34022#learn_the_details-evaluation).
