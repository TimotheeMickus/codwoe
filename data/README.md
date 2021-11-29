# Dataset access.

**The datasets are available at the following page: [https://codwoe.atilf.fr/](https://codwoe.atilf.fr/).**

# Dataset structure

This document details the structure of the JSON dataset file we provide. More information is available on the competition website: [link](https://competitions.codalab.org/competitions/34022#participate-get_data).


## Brief Overview

As an overview, the expected usage of the datasets is as follow:
 + In the Definition Modeling track, we expect participants to use the embeddings ("char", "sgns", "electra") to generate the associated definition ("gloss").
 + In the Reverse Dictionary track, we expect participants to use the definition ("gloss") to generate any of the associated embeddings ("char", "sgns", "electra").


## Dataset files structure

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

## Description of contents

The value associated to "id" tracks the language, data split and unique identifier for this example.

The value associated to the "gloss" key is a definition, as you would find in a classical dictionary. It is to be used either the target in the Definition Modeling track, or  asthe source in the Reverse Dictionary track.

All other keys ("char", "sgns", "electra") correspond to embeddings, and the associated values are arrays of floats representing the components. They all can serve as targets for the Reverse Dictionary track.
 + "char" corresponds to character-based embeddings, computed using an auto-encoder on the spelling of a word.
 + "sgns" corresponds to skip-gram with negative sampling embeddings (aka. word2vec)
 + "electra" corresponds to Transformer-based contextualized embeddings.


## Using the dataset files

Given that the data is in JSON format, it is straightforward to load it in python:

```python
import json
with open(PATH_TO_DATASET, "r") as file_handler:
    dataset = json.load(file_handler)
```

A more complete example for pytorch is available in the git repository (see here: [link](https://git.atilf.fr/tmickus/codwoe/-/blob/master/code/data.py#L18)).

## Expected output format

During the evaluation phase, we will expect submissions to reconstruct the same JSON format.

The test JSON files for input will be separate for each track. They will contain the "id" key, and either the "gloss" key (in the reverse dictionary track) or the embedding keys ("char" and "sgns" keys, and "electra" "key" in EN/FR/RU, in the definition modeling track).

In the definition modeling track, participants should construct JSON files that contain at least the two following keys:
 + the original "id"
 + their generated "gloss"

In the reverse dictionary, participants should construct JSON files that contain at least the two following keys:
 + the original "id",
 + any of the valid embeddings ("char", "sgns", or "electra" key in EN/FR/RU)

Other keys can be added. More details concerning the evaluation procedure are available here: [link](https://competitions.codalab.org/competitions/34022#learn_the_details-evaluation).

## License Information

The complete datasets, embedding architectures and embedding models will be made publicly available after the evaluation phase under a CC-BY-SA license. Please link to the competition website page ([link](https://competitions.codalab.org/competitions/34022)) and cite our upcoming task description paper if you use these datasets in your own work.

Dictionary data has been extracted from dumps provided by [Sérasset (2014)](http://kaiko.getalp.org/about-dbnary/). Embeddings were trained specifically for this shared task; all details will be made available in the task description paper.

