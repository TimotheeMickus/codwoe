# Baseline scores

Here are baseline results on the development set for the two tracks, obtained with the architectures described in this sub-directory.
The code here is itself based on the baselines we provided earlier, along with a couple improvements:
- a principled way of selecting hyperparameters (using Bayesian Optimization),
- a sentence-piece retokenization, to ensure the vocabulary is of the same size for all languages,
- a beam-search decoding for the definition modeling pipeline.

## Installation
To train these models, you will also need need the `scikit-learn` and `scikit-optimize` libraries, which we used to select hyperparameters.
```sh
pip3 install scikit-learn==0.24.2 scikit-optimize==0.8.1
```


## Scores
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
| en SGNS    | 0.00125    | 0.00250    | 0.10339
| en char    | 0.00011    | 0.00022    | 0.08852
| en electra | 0.00165    | 0.00215    | 0.08798
| es SGNS    | 0.01536    | 0.02667    | 0.20130
| es char    | 0.01505    | 0.02471    | 0.19933
| fr SGNS    | 0.00351    | 0.00604    | 0.18478
| fr char    | 0.00280    | 0.00706    | 0.18579
| fr electra | 0.00219    | 0.00301    | 0.17391
| it SGNS    | 0.02591    | 0.04081    | 0.20527
| it char    | 0.00640    | 0.00919    | 0.15902
| ru SGNS    | 0.01520    | 0.02112    | 0.34716
| ru char    | 0.01313    | 0.01847    | 0.32307
| ru electra | 0.01189    | 0.01457    | 0.33577
