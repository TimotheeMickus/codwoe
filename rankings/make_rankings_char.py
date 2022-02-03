import pandas as pd

METRICS = {
	'MSE':False,
	'cos':True,
	'rank':False,
}
LANGS = ['EN', 'ES', 'FR', 'IT', 'RU']

df = pd.read_csv('submission_scores/res_revdict-char.csv')
def get_sorted_vals(colname):
	return sorted(df[colname].dropna(), reverse=False)
for lang, metric in [(lang, metric) for lang in LANGS for metric in METRICS]:
	colname = f"{lang} {metric}"
	sorted_vals = get_sorted_vals(colname)
	to_maximize = METRICS[metric]
	def float_to_rank(cell):
		if pd.isna(cell):
			return cell
		if to_maximize:
			return sum(i >= cell  for i in sorted_vals)
		return sum(i <= cell  for i in sorted_vals)

	df[colname] = df[colname].apply(float_to_rank)
df.to_csv('submission_ranks/results_revdict-char-rankings.csv', index=False)
df_ranks = df.groupby('user').min()
for lang in LANGS:
	def get_mean_rank(row):
		metrics = [row[f"{lang} {metric}"] for metric in METRICS]
		if any(map(pd.isna, metrics)): return pd.NA
		return sum(metrics) / len(metrics)
	df_ranks[f"Rank {lang}"] = df_ranks.apply(get_mean_rank, axis=1)
del df_ranks['Date']
del df_ranks['filename']
df_ranks.to_csv('final_rankings/revdict-char_rankings-per-users.csv')
