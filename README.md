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

Will be added to this repository at a later date: data or link to the data for
the competition, code for training our models, and official results.

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

To get involved: check out the [codalab competition](http://example.org/). There
is also a participants'
["semeval2022-dictionaries-and-word-embeddings" google group](mailto:semeval2022-dictionaries-and-word-embeddings@googlegroups.com).
You can reach us organizers through [this email](mailto:tmickus@atilf.fr); make
sure to mention SemEval in your email object.

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
