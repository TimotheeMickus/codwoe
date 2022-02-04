# What is in this directory?

This subdirectory contains three subdirectories.
First is `submission_scores`, which lists the submissions we received and the scores they were attributed by the scoring program.
Second is `submission_ranks`, which converts submission scores into ranks (i.e., how many submissions fared better or equal to this one?).
Last is `final_rankings`, which lists the maximum rank per user, the average of these maximum ranks, and a nominal ranking per users, as listed below.

We also include the python scripts we used to convert raw submission scores into official rankings.

# Official rankings

Below are the official rankings for the SemEval 2022 CODWOE Shared task.

### Definition Modeling track

Below are the results for the Definition Modeling track.

| user / team   | Rank EN | Rank ES | Rank FR | Rank IT | Rank RU
|---------------|--------:|--------:|--------:|--------:|--------:
| Locchi        | 8       | 6       |         | 7       |
| WENGSYX       | 9       | 7       | 6       | 6       | 6
| cunliang.kong | 3       | 2       | 3       | **1**   | 2
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
| Nihed_Bendahman_ | 5       | 5       | 4       | 6       | 4
| WENGSYX          | **1**   | 2       | 2       | 3       | **1**   
| aardoiz          |         | 3       |         |         |
| chlrbgus321      | N/A     |         |         |         |
| IRB-NLP          | 3       | **1**   | **1**   | **1**   | 2
| pzchen           | 2       | 4       | 3       | 2       | 3
| the0ne           | 7       |         |         |         |
| tthhanh          | 8       | 7       | 6       | 7       | 6
| zhwa3087         | 6       | 6       | 5       | 5       | 5

#### B. ELECTRA targets

| user / team      | Rank EN | Rank FR | Rank RU
|------------------|--------:|--------:|--------:
| Locchi           | 3       |         |
| Nihed_Bendahman_ | 2       | 2       | 4
| WENGSYX          | 4       | 4       | 2
| IRB-NLP          | 5       | 3       | 3
| pzchen           | **1**   | **1**   | **1**   
| the0ne           | 6       |         |


#### C. Char-based targets

| user / team      | Rank EN | Rank ES | Rank FR | Rank IT | Rank RU
|------------------|--------:|--------:|--------:|--------:|--------:
| Locchi           | **1**   |         |         | 4       |
| Nihed_Bendahman_ | 2       | 2       | 2       | 3       | 4
| WENGSYX          | 7       | 5       | 5       | 6       | 5
| IRB-NLP          | 4       | 3       | 4       | 2       | 2
| pzchen           | 3       | **1**   | **1**   | **1**   | **1**   
| the0ne           | 5       |         |         |         |
| zhwa3087         | 6       | 4       | 3       | 5       | 3


# How were rankings computed?

See the python scripts.

We start by converting scalar scores into ranked scores: i.e., instead of considering the absolute value obtained by a submission, we count how many submissions fared better than the current one.
This is done so as to neutralize the fact that some metrics have to be maximized (e.g., cosine), whereas others ought to be minimized (e.g., MSE), and that all metrics have different maxima and minima (e.g., MSE is defined from 0 to infinity, but cosine is defined from -1 to +1).

We then take the maximum rank per user, and average across the three metrics for each target, namely:
 - MSE, cosine and rank-cosine for each target architecture in the Reverse Dictionary track
 - S-BLEU, L-BLEU and MoverScore for the Definition Modeling

Finally, we manually converted the average per targets into nominal rankings.
