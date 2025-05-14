import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

results_folder = '/local/home/rwiersma/Dev/Adobe/svbrdf_uncertainty/paper_experiments/out/sh_synthetic/'

def read_results(folder):
    results = pd.DataFrame()
    for root, dirs, files in os.walk(folder):
        if 'dropout_table.csv' not in files:
            continue
        df = pd.read_csv(os.path.join(root, 'dropout_table.csv'))
        results = pd.concat([results, df], ignore_index=True)

    # Sort results by n_views
    results = results.sort_values(by=['n_views'])
    # Compute the psnr from the mse
    results['psnr'] = results['mse'].apply(lambda x: -10 * np.log10(x))

    # Compute gain in PSNR per shape for every n_views, compared to n_views=10
    baseline_psnr = results[results['n_views'] == 10].set_index('shape')['psnr']
    results['gain'] = results.apply(lambda row: row['psnr'] - baseline_psnr[row['shape']], axis=1)
    return results

results_best = read_results(results_folder + 'dropout_best_view')
results_even = read_results(results_folder + 'dropout_evenly_spaced')

t_test_result = stats.ttest_rel(results_best['gain'], results_even['gain'])
print(t_test_result)

fontManager.addfont(path='/local/home/rwiersma/.local/share/fonts/LinBiolinum_R.otf')

plt.rcParams["font.family"] = 'Linux Biolinum O'
plt.rcParams["font.size"] = 12

red = '#F15E46'
blue = '#3DAEE3'
purple = '#8842eb'
grey = '#BBBBBB'

# Data to plot
n_points = results_even['n_views'].unique()
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# PSNR Gain
mean_gain_even = results_even.groupby('n_views')['gain'].mean().values
mean_gain_best = results_best.groupby('n_views')['gain'].mean().values
ax[0].plot(n_points, mean_gain_even, label='Even', c=red, linestyle='--')
ax[0].annotate('Even', xy= (n_points[-2], mean_gain_even[-2] - 0.1), color=red, fontsize=14)
ax[0].plot(n_points, mean_gain_best, label='Best', c=red)
ax[0].annotate('Best', xy= (n_points[-2], mean_gain_best[-2] - 0.1), color=red, fontsize=14)

# Entropy
mean_entropy_even = results_even.groupby('n_views')['entropy'].mean().values
mean_entropy_best = results_best.groupby('n_views')['entropy'].mean().values
ax[1].plot(n_points, mean_entropy_even, label='Even', c=purple, linestyle='--')
ax[1].annotate('Even', xy= (n_points[-2], mean_entropy_even[-2] + 0.01), color=purple, fontsize=14)
ax[1].plot(n_points, mean_entropy_best, label='Best', c=purple)
ax[1].annotate('Best', xy= (n_points[-2], mean_entropy_best[-2] + 0.01), color=purple, fontsize=14)

ax[0].set_ylabel('PSNR gain')
ax[1].set_ylabel('Average Entropy')
ax[0].set_xlabel('Number of views')
ax[1].set_xlabel('Number of views')

for i in (0, 1):
    ax[i].grid(which='major', axis='both', linestyle='-', color='grey', linewidth=0.5)
    ax[i].set_facecolor((0.95, 0.95, 0.95))
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['bottom'].set_visible(False)
    ax[i].spines['left'].set_visible(False)
    ax[i].grid(visible=True, axis='y', c='white', linewidth=1)
    ax[i].grid(visible=False, axis='x')

plt.savefig('out/application_capture.pdf', format='pdf', bbox_inches='tight')
plt.show()