import pandas as pd
METRICS = ['MvSc.', 'S-BLEU', 'L-BLEU']
LANGS = ['EN', 'ES', 'FR', 'IT', 'RU']
df = pd.read_csv('res_defmod.csv')
def get_sorted_vals(colname):
	return sorted(df[colname].dropna(), reverse=True)
for colname in [f"{lang} {metric}" for lang in LANGS for metric in METRICS]:
	sorted_vals = get_sorted_vals(colname)
	def float_to_rank(cell):
		if pd.isna(cell): return cell
		return sum(i >= cell  for i in sorted_vals)
	df[colname] = df[colname].apply(float_to_rank)
df.to_csv('results_defmod.csv', index=False)
df_ranks = df.groupby('user').min()
for lang in LANGS:
	def get_mean_rank(row):
		metrics = [row[f"{lang} {metric}"] for metric in METRICS]
		if any(map(pd.isna, metrics)): return pd.NA
		return sum(metrics) / len(metrics)
	df_ranks[f"Rank {lang}"] = df_ranks.apply(get_mean_rank, axis=1)
del df_ranks['Date']
del df_ranks['filename']
df_ranks.to_csv('defmod_rankings-per-users.csv')
